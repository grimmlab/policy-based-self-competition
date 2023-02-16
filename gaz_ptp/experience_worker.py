import os
import copy
import random
import sys
import psutil

import torch
import numpy as np
import ray
import time
from base_config import BaseConfig
from base_game import BaseGame
from local_inferencer import LocalInferencer
from model.base_network import BaseNetwork
from gumbel_mcts import GumbelMCTS, GumbelNode
from inferencer import ModelInferencer
from shared_storage import SharedStorage

from typing import Dict, Optional, Type, Union
from typing_extensions import TypedDict


class GameOptions(TypedDict):
    # Whether to use MCTS when choosing a move. Specified for each player (e.g. {1: True, -1: False})
    # If one player does not use MCTS, then the opponent also simulates the other player
    # as not acting accordingly to GAZ (rather samples actions straight from predicted policy).
    use_tree_simulations_for_players: Dict[int, bool]
    # Which models to use for each player when performing inference of states,
    # by specifying the model key, i.e. "newcomer" or "best". Specified for each player (e.g. {1: "newcomer", -1:"best})
    model_for_players: Dict[int, str]
    # If moves should be selected deterministically from the priors. Specified for each player. Is ignored if player
    # uses GAZ MCTS (as action from sequential halving is chosen).
    select_moves_deterministically: Dict[int, bool]
    # Within MCTS, if moves should be simulated deterministically. Specified for each player.
    # Note that even if one player does not use MCTS, this player's moves will be simulated as specified here by the
    # opponent (i.e. moves are not sampled from predicted policy, but chosen greedily). Serves as switch for PTP GT and ST
    mcts_simulate_moves_deterministically: Dict[int, bool]
    # The baseline player is the one for which the baseline policy is used, in the paper referred to as "greedy actor".
    baseline_player: Optional[int]
    # If this is set to a player, this player simply plays its move by sampling uniformly random from
    # number of actions (use only for testing purposes).
    random_player: Optional[int]
    # Indicates whether the game stems from a self-play game or not (as opposed to self-competition).
    is_self_play: bool


@ray.remote
class ExperienceWorker:
    """
    GAZ PTP Experience Worker.
    Instances of this class run in separate processes and continuously play matches of GAZ PTP against themselves.
    The game history is saved to the global replay buffer, which is accessed by the training process which optimizes
    the networks.
    """
    def __init__(self, actor_id: int, config: BaseConfig, shared_storage: SharedStorage, model_inference_worker: Union[str, ModelInferencer],
                 game_class: Type[BaseGame], network_class: Type[BaseNetwork], random_seed: int = 42, cpu_core: Optional[int] = None):
        """
        actor_id [int]: Unique id to identify the self play process. Is used for querying the inference models, which
            send back the results to the actor.
        config [BaseConfig]: Config
        shared_storage [SharedStorage]: Shared storage worker.
        model_inference_worker [ModelInferencer]: Instance of model inferencer to which the actor sends states to evaluate.
            If a string is given, it is expected to be the device on which an instance of LocalInferencer should run.
        game_class: Subclass of BaseGame from which instances of games are constructed
        network_class: BaseNetwork subclass used in game.
        random_seed [int]: Random seed for this actor
        cpu_core [Optional[int]]: If not None, the actor is pinned to the core with given index. Helps preventing
            multiple numpy/torch processes to fight over resources.
        """
        self.actor_id = actor_id
        self.config = config

        # Pin workers to core with given index
        if config.pin_workers_to_core and sys.platform == "linux" and cpu_core is not None:
            os.sched_setaffinity(0, {cpu_core})
            psutil.Process().cpu_affinity([cpu_core])

        if self.config.CUDA_VISIBLE_DEVICES:
            # override ray's limiting of GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config.CUDA_VISIBLE_DEVICES

        if isinstance(model_inference_worker, str):
            model_inference_worker = LocalInferencer(
                config=self.config,
                shared_storage=shared_storage,
                network_class=network_class,
                model_named_keys=["newcomer", "best"],
                initial_checkpoint=None,  # is set in local inferencer
                device=torch.device(model_inference_worker)
            )

        self.model_inference_worker = model_inference_worker
        self.shared_storage = shared_storage
        self.game_class = game_class
        self.network_class = network_class
        self.n_games_played = 0

        # Stores MCTS tree which is persisted over the full game
        self.mcts = None
        # Set the random seed for the worker
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    def play_game(self, game_options: GameOptions, problem_instance=None):
        """
        Performs one match of the GAZ PTP game based on MCTS and sampling.
        """
        game_time = time.perf_counter()  # track how long the worker needs for a game
        # initialize game and game history
        if problem_instance is None:
            problem_instance = self.game_class.generate_random_instance(self.config.problem_specifics)

        winner_in_case_of_draw = 1  # we let player 1 win in case of draw to discourage copying of moves

        game: BaseGame = self.game_class(instance=problem_instance, winner_in_case_of_draw=winner_in_case_of_draw)
        game_history = GameHistory()

        game_history.observation_history.append(game.get_current_state())
        game_history.to_play_history.append(game.get_current_player())

        game_done = False

        game_stats = {
            "winner": 0,
            "objectives": {
                1: float("-inf"),
                -1: float("-inf")
            },
            "sequences": {  # full action sequences
                1: None,
                -1: None
            },
            "baseline_player": game_options["baseline_player"],
            # stores the probability distributions for selected moves for both baseline and newcomer
            "policies_for_selected_moves": {},
            "max_search_depth": 0,  # track how deep a single search simulation from the MCTS root goes at maximum
        }

        for n_moves in self.config.log_policies_for_moves:
            game_stats["policies_for_selected_moves"][n_moves] = {"newcomer": None, "baseline": None}

        move_counter = {
            1: 0,
            -1: 0
        }

        if not self.config.inference_on_experience_workers:
            ray.get(self.model_inference_worker.register_actor.remote(self.actor_id))
        with torch.no_grad():
            self.mcts = tree = GumbelMCTS(actor_id=self.actor_id, config=self.config,
                                          model_inference_worker=self.model_inference_worker, game_options=game_options,
                                          network_class=self.network_class)

            while not game_done:
                # run the simulation
                num_actions = len(game.get_actions())

                root, mcts_info = tree.run_at_root(game, num_actions)

                # Store maximum search depth for inspection
                if "max_search_depth" in mcts_info and game_stats["max_search_depth"] < mcts_info["max_search_depth"]:
                    game_stats["max_search_depth"] = mcts_info["max_search_depth"]

                if num_actions == 1:
                    # auto-choose single possible action
                    action = 0
                elif game_options["use_tree_simulations_for_players"][root.to_play]:
                    action = root.sequential_halving_chosen_action  # choose the action obtained through GAZ
                else:
                    # Action is simply chosen from prior
                    action = self.select_action_from_priors(
                        node=root,
                        deterministic=game_options["select_moves_deterministically"][root.to_play]
                    )
                # Make the chosen move
                game_done, reward = game.make_move(action)
                # store statistics in the history, as well as the next observations/player/level
                game_history.action_history.append(action)
                game_history.store_gumbel_search_statistics(tree, game_options, num_actions, self.config.gumbel_simple_loss)

                move_counter[root.to_play] += 1
                n_move = move_counter[root.to_play]
                # for the logger: store probability distribution
                if n_move in self.config.log_policies_for_moves:
                    policy = [child.prior for child in root.children.values()]
                    player = "newcomer" if root.to_play != game_options["baseline_player"] else "baseline"
                    game_stats["policies_for_selected_moves"][n_move][player] = policy

                # important: shift must happen after storing search statistics
                tree.shift(action)

                # store next observation and player
                game_history.observation_history.append(game.get_current_state())
                game_history.to_play_history.append(game.get_current_player())

                if game_done:
                    game_time = time.perf_counter() - game_time
                    winner = game.get_current_player() if reward == 1 else -1 * game.get_current_player()

                    game_stats["id"] = self.actor_id  # identify from which actor this game came from
                    game_stats["winner"] = winner
                    game_stats["objectives"][1] = game.get_objective(1)
                    game_stats["objectives"][-1] = game.get_objective(-1)
                    game_stats["sequences"][1] = game.get_sequence(1)
                    game_stats["sequences"][-1] = game.get_sequence(-1)
                    game_stats["game_time"] = game_time
                    game_stats["waiting_time"] = tree.waiting_time

        # Bootstrap winner from end of game directly through all states
        game_history.bootstrap_winner(winner=game_stats["winner"])
        game_history.is_self_play_game = game_options["is_self_play"]
        game_history.winner = game_stats["winner"]
        self.n_games_played += 1

        if not self.config.inference_on_experience_workers:
            ray.get(self.model_inference_worker.unregister_actor.remote(self.actor_id))

        return game_history, game_stats

    def add_query_results(self, results):
        self.mcts.add_query_results(results)

    def eval_mode(self):
        """
        In evaluation mode, data to evaluate is pulled from shared storage until evaluation mode is unlocked.
        """
        while ray.get(self.shared_storage.in_evaluation_mode.remote()):
            to_evaluate = ray.get(self.shared_storage.get_to_evaluate.remote())

            if to_evaluate is not None:
                # We have something to evaluate
                eval_index, instance, eval_type = to_evaluate

                if eval_type == "arena":
                    # Arena mode
                    best_plays_randomly = ray.get(self.shared_storage.get_info.remote("best_plays_randomly"))

                    random_player = None if not best_plays_randomly else -1

                    game_options: GameOptions = {
                        "use_tree_simulations_for_players": {1: False, -1: False},
                        "model_for_players": {1: "newcomer", -1: "best"},
                        "select_moves_deterministically": {1: True, -1: True if not best_plays_randomly else False},
                        "mcts_simulate_moves_deterministically": {1: False, -1: False},  # irrelevant
                        "baseline_player": -1,  # irrelevant
                        "random_player": random_player,
                        "is_self_play": False
                    }
                    _, game_stats = self.play_game(
                        game_options=game_options,
                        problem_instance=instance
                    )

                    self.shared_storage.push_evaluation_result.remote((eval_index, copy.deepcopy(game_stats)))
                elif eval_type == "test":
                    # Testing environment. Player 1 is our current model.
                    use_tree = not self.config.gumbel_test_greedy_rollout
                    game_options: GameOptions = {
                        "use_tree_simulations_for_players": {1: use_tree, -1: False},
                        "model_for_players": {1: "newcomer", -1: "best"},
                        "select_moves_deterministically": {1: True, -1: True},
                        "mcts_simulate_moves_deterministically": {1: True, -1: True},
                        "baseline_player": -1,
                        "random_player": None,
                        "is_self_play": False
                    }
                    _, game_stats = self.play_game(
                        game_options=game_options,
                        problem_instance=instance
                    )
                    self.shared_storage.push_evaluation_result.remote((eval_index, copy.deepcopy(game_stats)))

                else:
                    raise ValueError(f"Unknown eval_type {eval_type}.")
            else:
                time.sleep(1)

    def continuous_play(self, replay_buffer, logger=None):
        while not ray.get(self.shared_storage.get_info.remote("terminate")):

            if ray.get(self.shared_storage.in_evaluation_mode.remote()):
                self.eval_mode()

            # Play a regular game for filling the replay buffer.
            # We randomly choose the player for the "newcomer" model (= learning actor in paper)
            # trying to learn a dominating strategy.
            learning_player: int = random.choice([-1, 1])
            adversary_player: int = - learning_player

            infos: Dict = ray.get(self.shared_storage.get_info.remote(["best_plays_randomly", "num_played_games"]))
            best_plays_randomly = infos["best_plays_randomly"]
            num_played_games = infos["num_played_games"]

            self_play_parameter = self.config.initial_self_play_parameter \
                if num_played_games < self.config.reduce_self_play_parameter_after_n_games \
                else self.config.self_play_parameter

            # With a certain probability, "best" (= greedy actor) plays with the current newcomer model
            learning_model = "newcomer"
            adversary_model = "best"
            is_self_play = False
            is_gt = self.config.gumbel_is_gt
            if np.random.rand() < self_play_parameter:
                adversary_model = "newcomer"
                is_self_play = True

            game_options: GameOptions = {
                "use_tree_simulations_for_players": {learning_player: True, adversary_player: False},
                "model_for_players": {learning_player: learning_model, adversary_player: adversary_model},
                "select_moves_deterministically": {learning_player: True, adversary_player: True if not best_plays_randomly else False},
                "mcts_simulate_moves_deterministically": {learning_player: False, adversary_player: is_gt},
                "baseline_player": adversary_player,
                "random_player": None if not best_plays_randomly else adversary_player,
                "is_self_play": is_self_play
            }

            game_history, game_stats = self.play_game(
                game_options=game_options,
                problem_instance=None
            )

            game_history.learning_player = learning_player
            game_stats["newcomer"] = learning_player

            # save game to the replay buffer and notify logger
            replay_buffer.save_game.remote(game_history, self.shared_storage)
            if logger is not None:
                logger.played_game.remote(game_stats, "train")

            if self.config.ratio_range:
                infos: Dict = ray.get(
                    self.shared_storage.get_info.remote(["training_step", "num_played_games", "terminate"]))
                num_played_games = infos["num_played_games"]
                num_games_in_replay_buffer = ray.get(replay_buffer.get_length.remote())
                ratio = infos["training_step"] / max(1, num_played_games - self.config.start_train_after_episodes)

                while (ratio < self.config.ratio_range[0] and num_games_in_replay_buffer > self.config.start_train_after_episodes
                       and not infos["terminate"] and not ray.get(self.shared_storage.in_evaluation_mode.remote())
                ):
                    infos: Dict = ray.get(
                        self.shared_storage.get_info.remote(["training_step", "num_played_games", "terminate"])
                    )
                    num_games_in_replay_buffer = ray.get(replay_buffer.get_length.remote())
                    ratio = infos["training_step"] / max(1, infos["num_played_games"] - self.config.start_train_after_episodes)
                    time.sleep(0.010)  # wait for 10ms

    def select_action_from_priors(self, node: GumbelNode, deterministic: bool) -> int:
        """
        Select discrete action according to visit count distribution or directly from the node's prior probabilities.
        """
        priors = [child.prior for child in node.children.values()]
        actions = [action for action in node.children.keys()]

        if deterministic:
            return actions[priors.index(max(priors))]

        return np.random.choice(actions, p=priors)


class GameHistory:
    """
    Stores information about the moves in a game.
    """
    def __init__(self):
        # Observation is a np.array containing the state of the env of the current player and the waiting player
        self.observation_history = []
        # i-th entry corresponds to the action the player took who was on move in i-th observation. For simultaenous
        # this is a tuple of actions.
        self.action_history = []
        # i-th entry corresponds to reward from the game (who lost who won) from the perspective of the player in i-th
        # observation. Bootstrapped directly from end of the game!
        # For simultaneous move this is from perspective of player 1.
        self.value_history = []
        # player on move in i-th observation
        self.to_play_history = []
        # stores the value of the root node at i-th observation after the tree search.
        self.root_values = []
        # stores the action policy of the root node at i-th observation after the tree search, depending on visit
        # counts of children. Each element is a list of length number of actions on level, and sums to 1.
        self.root_policies = []

        # Stores the learning player, i.e. the player which tries to improve her policy. Used for sampling in replay
        # buffer. This value is set to "0" in the beginning to indicate it is unset and is
        # defined by `continuous_self_play`.
        self.learning_player = 0

        self.is_self_play_game = False

        self.winner = 0

    def store_gumbel_search_statistics(self, mcts: GumbelMCTS, game_options: GameOptions, num_actions: int,
                                       for_simple_loss: bool = False):
        """
        Stores the improved policy of the root node.
        """
        root = mcts.root
        if not game_options["use_tree_simulations_for_players"][root.to_play] or num_actions == 1:
            # We have no visit counts for this node's actions, so just use the priors
            policy = [child.prior for child in root.children.values()]
        else:
            if for_simple_loss:
                # simple loss is where we assign probability one to the chosen action
                action = root.sequential_halving_chosen_action
                policy = [0.] * len(root.children)
                policy[action] = 1.
            else:
                policy = mcts.get_improved_policy(root).numpy().tolist()

        self.root_values.append(root.value())
        self.root_policies.append(policy)

    def bootstrap_winner(self, winner: int):
        """
        Iterate over all states and add a true value to the states, which is "1" if the player on move
        won in the end, else "-1".

        Parameters:
            winner [int]: Winner of game.
        """
        for to_play in self.to_play_history:
            self.value_history.append(1 if to_play == winner else -1)
