import copy
import random
import sys
import psutil
import os

import torch
import numpy as np
import ray
import time
from base_config import BaseConfig
from base_game import BaseGame
from local_inferencer import LocalInferencer
from model.base_network import BaseNetwork
from gumbel_mcts_single import SingleplayerGumbelMCTS, SingleplayerGumbelNode
from inferencer import ModelInferencer
from shared_storage import SharedStorage

from typing import Dict, Optional, Type, Union
from typing_extensions import TypedDict


class RolloutValueEstimator:
    def __init__(self, config: BaseConfig, shared_storage: SharedStorage,
                 game_class: Type[BaseGame], network_class: Type[BaseNetwork]):
        self.config = config
        self.inferencer = LocalInferencer(
                config=self.config,
                shared_storage=shared_storage,
                network_class=network_class,
                model_named_keys=["newcomer", "best"],
                initial_checkpoint=None,  # is set in CPU Inferencer
                device=torch.device("cpu")
            )

        self.game_class = game_class

    def get_timestep_value_estimation(self, instance, greedy: bool) -> np.array:
        value_estimate = []  # stores value estimates at each timestep, starting from state s_0
        game: BaseGame = self.game_class(instance=instance, baseline_outcome=0)
        game_done = False

        with torch.no_grad():
            while not game_done:
                # get policy for best and newcomer. Choose move by best, and value estimate by newcomer.
                state = game.get_current_state()
                num_actions = len(game.get_actions())
                policy_logits_padded, _ = self.inferencer.infer_batch(
                    [state],
                    "best"
                )
                _, value = self.inferencer.infer_batch(
                    [state],
                    "newcomer"
                )

                value_estimate.append(value[0][0].item())
                policy_logits = policy_logits_padded[0][:num_actions]

                if greedy:
                    action = np.argmax(policy_logits)
                else:
                    policy_probs = torch.softmax(torch.from_numpy(policy_logits), dim=0).numpy().astype('float64')
                    # normalize, in most cases the sum is not exactly equal to 1 which is problematic when sampling
                    policy_probs /= policy_probs.sum()
                    action = np.random.choice(list(range(num_actions)), p=policy_probs)

                # Make the chosen move
                game_done, reward = game.make_move(action)

                if game_done:
                    value_estimate.append(reward)

        return np.array(value_estimate)