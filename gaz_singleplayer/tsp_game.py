import math

import numpy as np
from typing import Tuple, List, Dict
from base_game import BaseGame
import gaz_ptp.tsp_game as tsp_game


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
        instance [np.array]: Cities (2-dimensional vectors) combined as an np.array in the shape (num_cities, 2).
            Each city lies in the unit square [0,1] x [0,1]
        baseline_outcome [float]: Greedy baseline against which to compete in this game.
        suppress_env_creation [bool]: If `True`, no environments are created. This is only used for custom copying of games.
            See `copy`-method below.
        """
        self.instance = instance
        self.n_cities = instance.shape[0]
        self.baseline_outcome = baseline_outcome

        if not suppress_env_creation:
            self.player_environment = tsp_game.TSPEnvironment(cities=self.instance)

        # holds the players' final schedule maketimes
        self.player_tour_length = float("inf")

        # flag indicating whether the game is finished
        self.game_is_over = False

    def get_current_player(self):
        return 1

    def get_objective(self, for_player: int) -> float:
        return self.player_tour_length

    def get_sequence(self, for_player: int) -> List[int]:
        return self.player_environment.current_tour

    def get_actions(self) -> List:
        """
        Legal actions for the current player is the number of remaining jobs. Indices 0, ..., <num remaining job> - 1
        will be mapped to the true jobs when making a move.
        """
        environment = self.player_environment
        return list(range(len(environment.remaining_nodes)))

    def is_finished_and_winner(self) -> Tuple[bool, int]:
        # Irrelevant for singeplayer games
        return self.game_is_over, 0

    def make_move(self, action: int):
        """
        Performs a move on the board, i.e. the player chooses the next city from its remaining cities.

        Parameters:
            action [int]: The index of the action to play, i.e. an index in 0, ..., (number of remaining nodes - 1)

        Returns:
            game_done [bool]: Boolean indicating whether the game is over.
            reward [int]: 0, if the game is unfinished. Otherwise, "1" if the player has a better objective than baseline, else "-1".
        """
        if self.game_is_over:
            raise Exception(f"Playing a move on a finished game!")

        current_tour_length, finished = self.player_environment.add_city_to_tour(remaining_node_idx=action)

        reward = 0.

        if finished:
            self.player_tour_length = current_tour_length
            if self.baseline_outcome:
                reward = -1 if self.player_tour_length >= self.baseline_outcome else 1
            else:
                reward = -1 * self.player_tour_length / (self.n_cities * math.sqrt(2))
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
        remaining_nodes, start_node, end_node, normalized_length, \
        start_end_node_distance_matrix, remaining_nodes_distance_matrix = \
            self.player_environment.get_current_situation()
        return [{
            "remaining_nodes": remaining_nodes,
            "start_node": start_node,
            "end_node": end_node,
            "tour_length": normalized_length,
            "start_end_node_distance_matrix": start_end_node_distance_matrix,
            "remaining_nodes_distance_matrix": remaining_nodes_distance_matrix,
            "baseline_outcome": self.baseline_outcome / (self.n_cities * math.sqrt(2))   # also normalize baseline
        }]

    def copy(self):
        """
        Way faster copy than deepcopy
        """
        game = Game(instance=self.instance, baseline_outcome=self.baseline_outcome, suppress_env_creation=True)
        game.player_environment = self.player_environment.copy()
        game.player_tour_length = self.player_tour_length
        game.game_is_over = self.game_is_over

        return game

    @staticmethod
    def generate_random_instance(problem_specific_config: Dict):
        return tsp_game.Game.generate_random_instance(problem_specific_config)

    @staticmethod
    def random_state_augmentation(states: List[Dict]):
        return tsp_game.BoardGeometry.random_state_augmentation(states)
