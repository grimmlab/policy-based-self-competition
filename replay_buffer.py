import copy
import collections

import numpy as np
import ray
import torch

from torch import nn
from base_config import BaseConfig
from base_game import BaseGame
from model.base_network import BaseNetwork

from typing import Dict, List, Type


@ray.remote
class ReplayBuffer:
    """
    Stores played episodes and generates batches for training the network.
    Runs in separate process, agents store their games in it asynchronously, while the
    trainer pulls batches from it.
    """

    def __init__(self, initial_checkpoint: Dict, config: BaseConfig, network_class: Type[BaseNetwork],
                 game_class: Type[BaseGame], prefilled_buffer: collections.deque = None):
        self.config = config
        self.network_class = network_class
        self.game_class = game_class
        # copy buffer if it has been provided
        if prefilled_buffer is not None:
            self.buffer = copy.deepcopy(prefilled_buffer)
        else:
            self.buffer = collections.deque([], maxlen=self.config.replay_buffer_size)

        self.num_played_games = initial_checkpoint["num_played_games"]
        self.num_played_steps = initial_checkpoint["num_played_steps"]

        # total samples keeps track of number of "available" total samples in the buffer (i.e. regarding only games
        # in buffer
        self.total_samples = sum(
            [len(game_history.root_values) for game_history in self.buffer]
        )

        if self.total_samples != 0:
            print(
                f"Replay buffer initialized with prefilled buffer: {self.total_samples} samples ({self.num_played_games} games)"
            )

        # Fix random seed
        np.random.seed(self.config.seed)

    def save_game(self, game_history, shared_storage=None):
        # Store an episode in the buffer.
        # As we are using `collections.deque, older entries get thrown out of the buffer
        self.num_played_games += 1
        self.num_played_steps += len(game_history.root_values)
        self.total_samples += len(game_history.root_values)

        if len(self.buffer) == self.config.replay_buffer_size:
            self.total_samples -= len(self.buffer[0].root_values)

        self.buffer.append(game_history)

        if shared_storage is not None:
            shared_storage.set_info.remote("num_played_games", self.num_played_games)
            shared_storage.set_info.remote("num_played_steps", self.num_played_steps)

        return self.num_played_games, self.num_played_steps, self.total_samples

    def get_batch(self, for_value=False):

        observation_batch = []
        value_batch = []
        policy_batch = []

        game_histories = np.random.choice(self.buffer, size=self.config.batch_size)

        for batch_idx, game_history in enumerate(game_histories):
            if self.config.singleplayer_options is None:
                # two-player game
                # get a game position where it's the turn of the learning player for sequential moving
                possible_positions = [i for i in range(len(game_history.action_history))
                                      if ((game_history.to_play_history[i] == game_history.learning_player and len(
                        game_history.root_policies[i]) >= 2) or for_value)]
            else:
                possible_positions = [i for i in range(len(game_history.action_history))
                                      if (len(game_history.root_policies[i]) >= 2 or for_value)]

            game_position = np.random.choice(possible_positions)
            target_value, target_policy = self.make_target(game_history, game_position)

            canonical_board = copy.deepcopy(game_history.observation_history[game_position])

            if self.config.problem_specifics["training_apply_geometry_augmentation"]:
                canonical_board = self.game_class.random_state_augmentation(canonical_board)

            observation_batch.append(canonical_board)

            value_batch.append(target_value)
            policy_batch.append(target_policy)

        value_batch_tensor = torch.cat(value_batch, dim=0)

        board_batch = self.network_class.states_to_batch(observation_batch, config=self.config)
        pad_policies_to = board_batch["current"]["policy_padded_to"]
        # pad the policies to maximum length in batch
        policy_lengths = [policy.shape[1] for policy in policy_batch]
        policy_averaging_tensor = torch.tensor(policy_lengths).float().unsqueeze(-1)
        policy_batch_tensor = torch.cat([
            nn.functional.pad(policy, (0, pad_policies_to - policy.shape[1]), value=0)
            for policy in policy_batch
        ], dim=0)

        return (
            board_batch,  # List of canonical boards
            value_batch_tensor,  # (batch_size, 1)
            policy_batch_tensor,  # Padded policies of shape (batch_size, <max policy length in batch>)
            policy_averaging_tensor
        )

    def get_length(self):
        return len(self.buffer)

    def make_target(self, game_history, state_index: int):
        """
        Generates targets (value and policy) for each observation.

        Parameters
            game_history: Episode history
            state_index [int]: Position in game to sample
        Returns:
            target_value: Float Tensor of shape (1, 1)
            target_policy: Tensor of shape (1, policy length)
        """
        if self.config.singleplayer_options is not None and self.config.singleplayer_options["method"] == "greedy_scalar":
            value = game_history.game_outcome
        elif self.config.singleplayer_options is not None and self.config.singleplayer_options["method"] in ["single", "single_timestep"]:
            value = self.singleplayer_value(game_history, state_index)
        else:
            value = game_history.value_history[state_index]
        target_value = torch.FloatTensor([value]).unsqueeze(0)

        policy = copy.deepcopy(game_history.root_policies[state_index])
        target_policy = torch.FloatTensor(policy).unsqueeze(0)

        return target_value, target_policy

    def singleplayer_value(self, game_history, state_index: int):
        if self.config.singleplayer_options["bootstrap_final_objective"]:
            return game_history.game_outcome
        else:
            bootstrap_n_steps = self.config.singleplayer_options["bootstrap_n_steps"]
            # The value target is the discounted root value of the search tree td_steps into
            # the future, plus the discounted sum of all rewards until then.
            if bootstrap_n_steps == -1:
                # sum up rewards until end
                bootstrap_index = len(game_history.root_values)
            else:
                bootstrap_index = min(state_index + self.config.singleplayer_options["bootstrap_n_steps"], len(game_history.root_values))

            value = game_history.root_values[bootstrap_index] if bootstrap_index < len(game_history.root_values) else 0
            value += sum(game_history.reward_history[state_index: bootstrap_index])
            return value

    def get_buffer(self):
        return self.buffer
