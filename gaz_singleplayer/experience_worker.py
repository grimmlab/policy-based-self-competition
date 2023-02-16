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
from gaz_singleplayer.value_estimator import RolloutValueEstimator
from model.base_network import BaseNetwork
from gumbel_mcts_single import SingleplayerGumbelMCTS, SingleplayerGumbelNode
from inferencer import ModelInferencer
from shared_storage import SharedStorage

from typing import Dict, Optional, Type, Union
from typing_extensions import TypedDict


@ray.remote
class ExperienceWorker:
    """
    GAZ PTP Experience Worker.
    Instances of this class run in separate processes and continuously play singleplayer matches.
    The game history is saved to the global replay buffer, which is accessed by the training process which optimizes
    the networks.
    """

    def __init__(self, actor_id: int, config: BaseConfig, shared_storage: SharedStorage, model_inference_worker: Union[str, ModelInferencer],
                 game_class: Type[BaseGame], network_class: Type[BaseNetwork], random_seed: int = 42, cpu_core: int = None):
        """
        actor_id [int]: Unique id to identify the self play process. Is used for querying the inference models, which
            send back the results to the actor.
        config [BaseConfig]: Config
        shared_storage [SharedStorage]: Shared storage worker.
        model_inference_worker [ModelInferencer]: Instance of model inferencer to which the actor sends states to evaluate
        game_class: Subclass of BaseGame from which instances of games are constructed
        random_seed [int]: Random seed for this actor
        """
        self.actor_id = actor_id
        self.config = config

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
                initial_checkpoint=None,  # is set in CPU Inferencer
                device=torch.device(model_inference_worker)
            )

        self.model_inference_worker = model_inference_worker
        self.shared_storage = shared_storage
        self.game_class = game_class
        self.network_class = network_class
        self.n_games_played = 0

        self.rollout_value_estimator: Optional[RolloutValueEstimator] = None
        if self.config.singleplayer_options["method"] == "single_timestep":
            self.rollout_value_estimator = RolloutValueEstimator(
                config=self.config,
                shared_storage=self.shared_storage,
                game_class=self.game_class, network_class=self.network_class
            )

        # check if we are in a self-competitive setting where we need to take some baseline into account
        self.use_baseline = self.config.singleplayer_options is not None \
                            and self.config.singleplayer_options["method"] == "greedy_scalar"

        # Stores MCTS tree which is persisted over the full game
        self.mcts: Optional[SingleplayerGumbelMCTS] = None
        # Set the random seed for the worker
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    def get_baseline(self, instance,
                     greedy: bool = True, model_to_use: str = "best"):
        game: BaseGame = self.game_class(instance=instance, baseline_outcome=0)
        game_done = False

        # check if we should play uniformly random with the baseline
        random_baseline = False
        if model_to_use == "best":
            random_baseline = ray.get(self.shared_storage.get_info.remote("best_plays_randomly"))

        if random_baseline:
            while not game_done:
                action = np.random.choice(game.get_actions())
                game_done, reward = game.make_move(action)
                if game_done:
                    baseline = game.get_objective(0)
            return baseline

        with torch.no_grad():
            if not self.config.inference_on_experience_workers:
                ray.get(self.model_inference_worker.register_actor.remote(self.actor_id))
            self.mcts = tree = SingleplayerGumbelMCTS(actor_id=self.actor_id, config=self.config,
                                                      model_inference_worker=self.model_inference_worker,
                                                      deterministic=greedy, min_max_normalization=False,
                                                      model_to_use=model_to_use)

            while not game_done:
                # run the simulation
                num_jobs = len(game.get_actions())
                root, mcts_info = tree.run_at_root(game, num_jobs, only_expand_root=True)

                action = self.select_action_from_priors(
                    node=root,
                    deterministic=greedy
                )

                # Make the chosen move
                game_done, reward = game.make_move(action)
                tree.shift(action)

                if game_done:
                    baseline = game.get_objective(0)

            if not self.config.inference_on_experience_workers:
                ray.get(self.model_inference_worker.unregister_actor.remote(self.actor_id))
        return baseline

    def play_game(self, problem_instance=None, greedy: bool = False, model_to_use="newcomer"):
        """
        Performs one match of the singleplayer game based on singleplayer GAZ MCTS.
        """
        game_time = time.perf_counter()  # track how long the worker needs for a game
        # initialize game and game history
        if problem_instance is None:
            problem_instance = self.game_class.generate_random_instance(self.config.problem_specifics)

        # Get baseline
        baseline = self.get_baseline(problem_instance, greedy=True, model_to_use="best") if self.use_baseline else 0
        baseline_to_log = baseline

        # Get rollout value estimation
        value_estimates: Optional[np.array] = None
        if self.config.singleplayer_options["method"] == "single_timestep":
            greedy = not self.config.singleplayer_options["baseline"]["sample"]
            baseline_to_log = self.get_baseline(problem_instance, greedy=True, model_to_use="best")
            num_rollouts = 1 if greedy else self.config.singleplayer_options["baseline"]["num_trajectories"]
            for _ in range(num_rollouts):
                rollout_value_estimate = self.rollout_value_estimator.get_timestep_value_estimation(problem_instance, greedy)
                if value_estimates is None:
                    value_estimates = rollout_value_estimate
                else:
                    value_estimates += rollout_value_estimate

            if num_rollouts > 1:
                value_estimates /= num_rollouts

        game: BaseGame = self.game_class(instance=problem_instance, baseline_outcome=baseline)
        game_history = GameHistory()

        game_history.observation_history.append(game.get_current_state())

        game_done = False

        game_stats = {
            "objective": 0,
            "sequence": None,
            "max_search_depth": 0,
            "policies_for_selected_moves": {},
            "baseline_objective": baseline_to_log
        }

        move_counter = 0

        if not self.config.inference_on_experience_workers:
            ray.get(self.model_inference_worker.register_actor.remote(self.actor_id))
        with torch.no_grad():
            use_min_max_norm = True if self.config.singleplayer_options is not None \
                                       and self.config.singleplayer_options["method"] in ["single", "single_timestep"] else False

            self.mcts = tree = SingleplayerGumbelMCTS(actor_id=self.actor_id, config=self.config,
                                                      model_inference_worker=self.model_inference_worker, deterministic=greedy,
                                                      min_max_normalization=use_min_max_norm,
                                                      model_to_use=model_to_use, value_estimates=value_estimates)

            while not game_done:
                # run the simulation
                num_actions = len(game.get_actions())

                root, mcts_info = tree.run_at_root(game, num_actions)

                # Store maximum search depth for inspection
                if "max_search_depth" in mcts_info and game_stats["max_search_depth"] < mcts_info["max_search_depth"]:
                    game_stats["max_search_depth"] = mcts_info["max_search_depth"]

                if num_actions == 1:
                    action = 0
                else:
                    action = root.sequential_halving_chosen_action

                # Make the chosen move
                game_done, reward = game.make_move(action)
                # store statistics in the history, as well as the next observations/player/level
                game_history.action_history.append(action)
                game_history.reward_history.append(reward)
                game_history.store_gumbel_search_statistics(tree, self.config.gumbel_simple_loss)

                move_counter += 1
                if move_counter in self.config.log_policies_for_moves:
                    policy = [child.prior for child in root.children.values()]
                    game_stats["policies_for_selected_moves"][move_counter] = policy

                # important: shift must happen after storing search statistics
                tree.shift(action)

                # store next observation
                game_history.observation_history.append(game.get_current_state())

                if game_done:
                    game_history.game_outcome = reward
                    game_time = time.perf_counter() - game_time
                    game_stats["id"] = self.actor_id  # identify from which actor this game came from
                    game_stats["objective"] = game.get_objective(0)
                    game_stats["sequence"] = game.get_sequence(0)
                    game_stats["game_time"] = game_time
                    game_stats["waiting_time"] = tree.waiting_time

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
                    stats = {
                        "objectives": {1: 0., -1: 0.}  # 1 for newcomer, -1 for best
                    }
                    for i, model_type in [(1, "newcomer"), (-1, "best")]:
                        baseline = self.get_baseline(instance, True, model_type)
                        stats["objectives"][i] = baseline
                    self.shared_storage.push_evaluation_result.remote((eval_index, copy.deepcopy(stats)))
                elif eval_type == "test":
                    if self.config.gumbel_test_greedy_rollout:
                        if self.config.singleplayer_options["method"] == "single":
                            rollout = self.get_baseline(instance, True, "newcomer")
                            game_stats = {
                                "objective": rollout,
                                "greedy_rollout": rollout
                            }
                        else:
                            game_stats = {
                                "objective": self.get_baseline(instance, True, "newcomer"),
                                "baseline_objective": self.get_baseline(instance, True, "best")
                            }
                    else:
                        _, game_stats = self.play_game(
                            problem_instance=instance, greedy=True,
                            model_to_use="newcomer"
                        )
                        if self.config.singleplayer_options["method"] in ["single"]:
                            greedy_rollout = self.get_baseline(instance, True, "newcomer")
                            game_stats["greedy_rollout"] = greedy_rollout
                    self.shared_storage.push_evaluation_result.remote((eval_index, copy.deepcopy(game_stats)))

                else:
                    raise ValueError(f"Unknown eval_type {eval_type}.")
            else:
                time.sleep(1)

    def continuous_play(self, replay_buffer, logger=None):
        while not ray.get(self.shared_storage.get_info.remote("terminate")):

            if ray.get(self.shared_storage.in_evaluation_mode.remote()):
                self.eval_mode()

            game_history, game_stats = self.play_game(
                problem_instance=None,
                greedy=False,
                model_to_use="newcomer"
            )

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

    def select_action_from_priors(self, node: SingleplayerGumbelNode, deterministic: bool) -> int:
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
        # stores estimated values for root states obtained from tree search
        self.root_values = []
        # stores the action policy of the root node at i-th observation after the tree search, depending on visit
        # counts of children. Each element is a list of length number of actions on level, and sums to 1.
        self.root_policies = []
        self.reward_history = []
        self.game_outcome: Optional[float] = None  # stores the final objective

    def store_gumbel_search_statistics(self, mcts: SingleplayerGumbelMCTS, for_simple_loss: bool = False):
        """
        Stores the improved policy of the root node.
        """
        root = mcts.root
        if for_simple_loss:
            # simple loss is where we assign probability one to the chosen action
            action = root.sequential_halving_chosen_action
            policy = [0.] * len(root.children)
            policy[action] = 1.
        else:
            policy = mcts.get_improved_policy(root).numpy().tolist()

        self.root_values.append(root.value())
        self.root_policies.append(policy)
