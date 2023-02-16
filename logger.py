import os

import ray
import time
import json
from typing import Dict
from base_config import BaseConfig


@ray.remote
class Logger:
    def __init__(self, config: BaseConfig, shared_storage, inferencers, is_singleplayer: bool):
        self.config = config
        self.shared_storage = shared_storage
        self.is_singleplayer = is_singleplayer

        self.n_played_games = 0
        # Check number of games played before this run (if a training is resumed from some checkpoint)
        self.n_played_games_previous = ray.get(shared_storage.get_info.remote("num_played_games"))
        self.rolling_game_stats = None
        self.play_took_time = 0

        self.reset_rolling_game_stats()

        self.n_trained_steps = 0
        self.n_trained_steps_previous = ray.get(shared_storage.get_info.remote("training_step"))
        self.rolling_loss_stats = None
        self.reset_rolling_loss_stats()

        self.inferencers = inferencers

        self.file_log_path = os.path.join(self.config.results_path, "log.txt")
        if self.config.do_log_to_file:
            os.makedirs(self.config.results_path, exist_ok=True)

    def reset_rolling_game_stats(self):

        self.play_took_time = time.perf_counter()
        self.rolling_game_stats = {
            "max_policies_for_selected_moves": {},
            "max_search_depth": 0,
            "game_time": 0,
            "waiting_time": 0
        }

        if self.is_singleplayer:
            self.rolling_game_stats["objective"] = 0
            self.rolling_game_stats["baseline_objective"] = 0
        else:
            self.rolling_game_stats["objectives"] = {"newcomer": 0, "best": 0, "winner": 0}
            self.rolling_game_stats["num_wins"] = {"newcomer": 0, "best": 0}

        for n_actions in self.config.log_policies_for_moves:
            if self.is_singleplayer:
                self.rolling_game_stats["max_policies_for_selected_moves"][n_actions] = 0
            else:
                self.rolling_game_stats["max_policies_for_selected_moves"][n_actions] = {
                    "newcomer": 0,
                    "baseline": 0
                }

    def reset_rolling_loss_stats(self):
        self.rolling_loss_stats = {
            "loss": 0,
            "value_loss": 0,
            "policy_loss": 0
        }

    def played_game(self, game_stats: Dict, game_type="train"):
        """
        Notify logger of new played game. game_stats is a dict of the form
            {
            "winner": +1/-1,
            "tour_lengths": { 1: npv player 1, -1: npv player 2 }
            }
        """
        self.n_played_games += 1
        self.rolling_game_stats["game_time"] += game_stats["game_time"]
        self.rolling_game_stats["max_search_depth"] += game_stats["max_search_depth"]
        if "waiting_time" in game_stats:
            self.rolling_game_stats["waiting_time"] += game_stats["waiting_time"]

        if self.is_singleplayer:
            self.rolling_game_stats["objective"] += game_stats["objective"]
            self.rolling_game_stats["baseline_objective"] += game_stats["baseline_objective"]
        else:
            self.rolling_game_stats["objectives"]["winner"] += game_stats["objectives"][game_stats["winner"]]
            for i in [1, -1]:
                player = "newcomer" if i == game_stats["newcomer"] else "best"
                self.rolling_game_stats["objectives"][player] += game_stats["objectives"][i]
                if game_stats["winner"] == i:
                    self.rolling_game_stats["num_wins"][player] += 1

        for n_actions in self.rolling_game_stats["max_policies_for_selected_moves"].keys():
            if self.is_singleplayer:
                self.rolling_game_stats["max_policies_for_selected_moves"][n_actions] += \
                    max(game_stats["policies_for_selected_moves"][n_actions])
            else:
                self.rolling_game_stats["max_policies_for_selected_moves"][n_actions]["newcomer"] += \
                    max(game_stats["policies_for_selected_moves"][n_actions]["newcomer"])
                self.rolling_game_stats["max_policies_for_selected_moves"][n_actions]["baseline"] += \
                    max(game_stats["policies_for_selected_moves"][n_actions]["baseline"])

        if self.n_played_games % self.config.log_avg_stats_every_n_episodes == 0:
            games_took_time = time.perf_counter() - self.play_took_time
            print(f'Num played games total: {self.n_played_games}')
            print(f"Episodes took time {games_took_time} s")

            # Get time it took for models on average
            avg_model_inference_time = 0
            if not self.config.inference_on_experience_workers:
                keys = ["full", "batching", "model"]
                inferencer_times = []
                for inferencer in self.inferencers:
                    inferencer_times.append(ray.get(inferencer.get_time.remote()))
                for key in keys:
                    inf_time = 0
                    for inferencer_time in inferencer_times:
                        inf_time += inferencer_time[key]
                    avg_model_inference_time = inf_time / len(self.inferencers)
                    print(f"Avg. model inference time '{key}': {avg_model_inference_time}")

            if self.is_singleplayer:
                avg_objective = self.rolling_game_stats["objective"] / self.config.log_avg_stats_every_n_episodes
                avg_baseline_objective = self.rolling_game_stats[
                                           "baseline_objective"] / self.config.log_avg_stats_every_n_episodes
            else:
                avg_objective_newcomer = self.rolling_game_stats["objectives"][
                                             "newcomer"] / self.config.log_avg_stats_every_n_episodes
                avg_objective_best = self.rolling_game_stats["objectives"][
                                         "best"] / self.config.log_avg_stats_every_n_episodes
                avg_objective_winner = self.rolling_game_stats["objectives"][
                                           "winner"] / self.config.log_avg_stats_every_n_episodes
                # ratio of newcomer winning the game
                win_ratio_newcomer = self.rolling_game_stats["num_wins"][
                                         "newcomer"] / self.config.log_avg_stats_every_n_episodes

            # average maximum search depth of games
            avg_max_depth = self.rolling_game_stats["max_search_depth"] / self.config.log_avg_stats_every_n_episodes

            # Average maximum probability for selected moves
            for n_actions in self.config.log_policies_for_moves:
                if self.is_singleplayer:
                    self.rolling_game_stats["max_policies_for_selected_moves"][n_actions] /= self.config.log_avg_stats_every_n_episodes
                else:
                    self.rolling_game_stats["max_policies_for_selected_moves"][n_actions][
                        "newcomer"] /= self.config.log_avg_stats_every_n_episodes
                    self.rolling_game_stats["max_policies_for_selected_moves"][n_actions][
                        "baseline"] /= self.config.log_avg_stats_every_n_episodes

            avg_time_per_game = self.rolling_game_stats["game_time"] / self.config.log_avg_stats_every_n_episodes
            avg_waiting_time_per_game = self.rolling_game_stats[
                                            "waiting_time"] / self.config.log_avg_stats_every_n_episodes
            print(f"Average time per game: {avg_time_per_game}")
            print(f"Average waiting time per game: {avg_waiting_time_per_game}")
            print(f'Avg max search depth per move: {avg_max_depth:.1f}')
            if self.is_singleplayer:
                print(f'Avg objective: {avg_objective}')
                print(f'Avg baseline objective: {avg_baseline_objective}')
                metrics_to_log = {
                    "Avg objective": avg_objective,
                    "Avg baseline objective": avg_baseline_objective
                }
            else:
                print(f'Avg objective winner: {avg_objective_winner}')
                print(f'Avg objective newcomer: {avg_objective_newcomer}')
                print(f'Avg objective best: {avg_objective_best}')
                print(f'Win ratio newcomer: {win_ratio_newcomer:.2f}')
                metrics_to_log = {
                    "Avg objective winner": avg_objective_winner,
                    "Avg objective newcomer": avg_objective_newcomer,
                    "Avg objective best": avg_objective_best,
                    "Win ratio newcomer": win_ratio_newcomer,
                }

            metrics_to_log["Games time in secs"] = games_took_time
            metrics_to_log["Avg game time in secs"] = avg_time_per_game
            metrics_to_log["Avg Inferencer Time in secs"] = avg_model_inference_time
            metrics_to_log["Avg max search depth per move"] = avg_max_depth

            for n_actions in self.config.log_policies_for_moves:
                if self.is_singleplayer:
                    metrics_to_log[f"Max policy newcomer {n_actions}"] = \
                        self.rolling_game_stats["max_policies_for_selected_moves"][n_actions]
                else:
                    metrics_to_log[f"Max policy newcomer {n_actions}"] = \
                        self.rolling_game_stats["max_policies_for_selected_moves"][n_actions]["newcomer"]
                    metrics_to_log[f"Max policy baseline {n_actions}"] = \
                        self.rolling_game_stats["max_policies_for_selected_moves"][n_actions]["baseline"]

            self.reset_rolling_game_stats()

            if self.config.do_log_to_file:
                # Additional things for logging to file
                metrics_to_log["Total num played games"] = self.n_played_games
                metrics_to_log["Total num trained steps"] = self.n_trained_steps
                metrics_to_log["Timestamp in ms"] = int(time.time() * 1000)
                metrics_to_log["logtype"] = "played_game"

                with open(self.file_log_path, "a+") as f:
                    f.write(json.dumps(metrics_to_log))
                    f.write("\n")

    def training_step(self, loss_dict: Dict):
        """
        Notify logger of performed training step. loss_dict has keys "loss", "value_loss" and "policy_loss" (all floats)
        for a batch on which has been trained.
        """

        self.n_trained_steps += 1

        self.rolling_loss_stats["loss"] += loss_dict["loss"]
        self.rolling_loss_stats["value_loss"] += loss_dict["value_loss"]
        self.rolling_loss_stats["policy_loss"] += loss_dict["policy_loss"]

        if self.n_trained_steps % self.config.log_avg_loss_every_n_steps == 0:
            # Also get training_steps to played_steps ratio
            training_steps = ray.get(self.shared_storage.get_info.remote("training_step"))
            played_games = ray.get(self.shared_storage.get_info.remote("num_played_games"))
            avg_loss = self.rolling_loss_stats["loss"] / self.config.log_avg_loss_every_n_steps
            avg_value_loss = self.rolling_loss_stats["value_loss"] / self.config.log_avg_loss_every_n_steps
            avg_policy_loss = self.rolling_loss_stats["policy_loss"] / self.config.log_avg_loss_every_n_steps

            ratio_steps_games = training_steps/played_games

            print(f"Total number of training steps: {self.n_trained_steps}, "
                  f"Ratio training steps to played games: {ratio_steps_games:.2f}, "
                  f"Avg loss: {avg_loss}, Avg value Loss: {avg_value_loss}, "
                  f"Avg policy loss: {avg_policy_loss}")
            self.reset_rolling_loss_stats()

            metrics_to_log = {
                    "Ratio training steps to played games": ratio_steps_games,
                    "Avg loss": avg_loss,
                    "Avg value loss": avg_value_loss,
                    "Avg policy loss": avg_policy_loss
                }

            if self.config.do_log_to_file:
                # Additional things for logging to file
                metrics_to_log["Total num played games"] = self.n_played_games
                metrics_to_log["Total num trained steps"] = self.n_trained_steps
                metrics_to_log["Timestamp in ms"] = int(time.time() * 1000)
                metrics_to_log["logtype"] = "training_step"

                with open(self.file_log_path, "a+") as f:
                    f.write(json.dumps(metrics_to_log))
                    f.write("\n")

    def arena_step(self, stats_dict: Dict):
        ratio_win_newcomer = stats_dict["newcomer_num_wins"] / max(1, (
                    stats_dict["newcomer_num_wins"] + stats_dict["best_num_wins"]))
        print(
            f"Arena done. Num Newcomer / Best wins: {stats_dict['newcomer_num_wins']} / {stats_dict['best_num_wins']} "
            f"(ratio: {ratio_win_newcomer:.2f}). "
            f"Avg objective newcomer: {stats_dict['avg_objective_newcomer']}, "
            f"Avg objective best: {stats_dict['avg_objective_best']}, "
            f"Avg objective margin: {stats_dict['avg_objective_margin']}")

        metrics_to_log = {
            "Arena win ratio newcomer": ratio_win_newcomer,
            "Arena objective best": stats_dict['avg_objective_best'],
            "Arena objective newcomer": stats_dict['avg_objective_newcomer'],
            "Arena objective margin": stats_dict['avg_objective_margin']
        }

        if self.config.do_log_to_file:
            # Additional things for logging to file
            metrics_to_log["Total num played games"] = self.n_played_games
            metrics_to_log["Total num trained steps"] = self.n_trained_steps
            metrics_to_log["Timestamp in ms"] = int(time.time() * 1000)
            metrics_to_log["logtype"] = "arena"

            with open(self.file_log_path, "a+") as f:
                f.write(json.dumps(metrics_to_log))
                f.write("\n")

    def evaluation_run(self, stats_dict: Dict):
        print(f"EVALUATION. Average objective: {stats_dict['avg_objective']}")

        if self.config.do_log_to_file:
            # Additional things for logging to file
            metrics_to_log = {
                "Total num played games": self.n_played_games,
                "Total num trained steps": self.n_trained_steps,
                "Timestamp in ms": int(time.time() * 1000),
                "logtype": "evaluation",
                "Evaluation Type": stats_dict['type'],
                "Evaluation Value": stats_dict['avg_objective']
            }

            with open(self.file_log_path, "a+") as f:
                f.write(json.dumps(metrics_to_log))
                f.write("\n")