import numpy as np
from typing import Tuple, List, Dict
from base_game import BaseGame
import gaz_ptp.jssp_game as jssp_game


class Game(BaseGame):
    """
    Class which poses the job shop scheduling problem as a singleplayer game against a scalar baseline.
    Actions are represented as integers (choosing a remaining job to schedule the next operation from).
    """
    def __init__(self, instance: np.array, baseline_outcome: float, suppress_env_creation: bool = False):
        """
        Initializes a new singleplayer game.

        Parameters
        ----------
        instance [np.array]: JSP Problem instance
        baseline_outcome [float]: Greedy baseline against which to compete in this game.
        suppress_env_creation [bool]: If `True`, no environments are created. This is only used for custom copying of games.
            See `copy`-method below.
        """
        self.instance = instance
        self.baseline_outcome = baseline_outcome

        if not suppress_env_creation:
            self.player_environment = jssp_game.JSSPEnvironment(instance=self.instance)

        # holds the players' final schedule maketimes
        self.player_maketime = float("inf")

        # flag indicating whether the game is finished
        self.game_is_over = False

    def get_current_player(self):
        return 1

    def get_objective(self, for_player: int) -> float:
        return self.player_maketime

    def get_sequence(self, for_player: int) -> List[int]:
        return self.player_environment.job_schedule_sequence

    def get_actions(self) -> List:
        """
        Legal actions for the current player is the number of remaining jobs. Indices 0, ..., <num remaining job> - 1
        will be mapped to the true jobs when making a move.
        """
        environment = self.player_environment
        return list(range(len(environment.remaining_job_idcs)))

    def is_finished_and_winner(self) -> Tuple[bool, int]:
        # Irrelevant for singeplayer games
        return self.game_is_over, 0

    def get_normalized_maketime(self):
        """
        For simplicity, in singleplayer games we scale the episodic reward to a normalized element in approximately [0,1],
        which is applicable for all considered JSSP sizes.
        """
        return self.player_maketime / 100.

    def make_move(self, action: int):
        """
        Performs a move on the board, i.e. the current player chooses the next job from its remaining jobs.

        Parameters:
            action [int]: The index of the action to play, i.e. an index in 0, ..., (number of remaining jobs - 1)

        Returns:
            game_done [bool]: Boolean indicating whether the game is over.
            reward [int]: 0, if the game is unfinished. Otherwise, "1" if the player has a better objective than baseline, else "-1".
        """
        if self.game_is_over:
            raise Exception(f"Playing a move on a finished game!")

        current_maketime, finished = self.player_environment.add_remaining_job_idx_to_schedule(remaining_job_idx=action)

        reward = 0.

        if finished:
            self.player_maketime = current_maketime
            if self.baseline_outcome:
                reward = -1 if self.player_maketime >= self.baseline_outcome else 1
            else:
                reward = -1 * self.get_normalized_maketime()
            self.game_is_over = True
            return True, reward

        return False, reward

    def get_current_state(self):
        """
        Returns the current singleplayer game situation.

        Returns:
            [List[Dict]] A list with a single element, which is a dictionary
                containing the situation of the player.
        """
        ret_list = []
        (
            operation_machine_idcs,
            operation_processing_times,
            num_remaining_jobs,
            job_availability_tensor,
            min_machine_time,
            machine_availability_tensor,
            num_ops_per_job
        ) = self.player_environment.get_current_continuous_situation()
        ret_list.append({
            "operation_machine_idcs": operation_machine_idcs,
            "operation_processing_times": operation_processing_times,
            "num_remaining_jobs": num_remaining_jobs,
            "job_availability_tensor": job_availability_tensor,
            "machine_availability_tensor": machine_availability_tensor,
            "min_machine_time": min_machine_time,
            "num_ops_per_job": num_ops_per_job,
            "baseline_outcome": self.baseline_outcome
        })

        return ret_list

    def copy(self):
        """
        Way faster copy than deepcopy
        """
        game = Game(instance=self.instance, baseline_outcome=self.baseline_outcome, suppress_env_creation=True)
        game.player_environment = self.player_environment.copy()
        game.player_maketime = self.player_maketime
        game.game_is_over = self.game_is_over

        return game

    @staticmethod
    def generate_random_instance(problem_specific_config: Dict):
        return jssp_game.Game.generate_random_instance(problem_specific_config)

    @staticmethod
    def random_state_augmentation(states: List[Dict]):
        return jssp_game.BoardGeometry.random_state_augmentation(states)

