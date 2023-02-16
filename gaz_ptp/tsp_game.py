import copy
import math
import random
import torch
import numpy as np
from typing import Tuple, List, Dict
from base_game import BaseGame


class TSPEnvironment:
    """
    Environment for one player, representing the current tour.
    """

    def __init__(self, cities: np.array, create_distance_matrix=True):
        """
        Parameters
        ----------
        cities [np.array]: Cities (2-dimensional vectors) combined as an np.array in the shape (num_cities, 2). Each city
            lies in the unit square [0,1] x [0,1]
        create_distance_matrix [bool]: If `False`, no distance matrix is created from the cities. This is only relevant
            for the class's internal `copy` method.
        """
        self.cities = cities
        self.n_cities = self.cities.shape[0]

        # (n_cities x n_cities) symmetric matrix containing the distances of cities to each other
        # This will be used as an attention bias in the Transformer architecture
        self.distance_matrix = self.create_distance_matrix() if create_distance_matrix else None

        # stores the tour and length so far. Cities in tour are indicated by indices of cities 0, ..., n_cities - 1
        self.current_tour: List[int] = []
        self.current_tour_length: float = 0

        # Stores indices of nodes which can be chosen
        self.remaining_nodes: List[int] = list(range(self.n_cities))

        # index of current position in the tour, from which the next node must be chosen
        self.start_node: int = -1
        # index of target position of the tour, is the same as the first node
        self.end_node: int = -1

        # indicates whether tour is finished or not
        self.finished = False

    def create_distance_matrix(self):
        distance_matrix = np.empty((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            distance_matrix[i, i] = 0
            for j in range(i):
                distance = np.sqrt(np.sum((self.cities[i] - self.cities[j]) ** 2))
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        return distance_matrix

    def copy(self):
        env = TSPEnvironment(cities=np.copy(self.cities), create_distance_matrix=False)
        env.distance_matrix = np.copy(self.distance_matrix)
        env.current_tour = self.current_tour.copy()
        env.current_tour_length = self.current_tour_length
        env.remaining_nodes = self.remaining_nodes.copy()
        env.start_node = self.start_node
        env.end_node = self.end_node
        env.finished = self.finished

        return env

    def add_city_to_tour(self, remaining_node_idx: int) -> Tuple[float, bool]:
        """
        Adds a city to the current tour. Automatically finalizes the tour if only one city is left (no alternatives).
        Parameters:
            remaining_node_idx [int]: Index of city in the remaining nodes. Note that this is not equal to the index
                of all cities. Example: If remaining nodes are [1,3,5], then `remaining_node_idx` 1 maps to
                city with index `3`.
        Returns:
            [float] Length of current tour.
            [bool] `True` if the tour is complete, else `False`.
        """
        if remaining_node_idx > len(self.remaining_nodes):
            raise Exception(f"Chosen remaining node idx {remaining_node_idx} "
                            f"city cannot be added to tour, as it is not available.")

        city_idx = self.remaining_nodes[remaining_node_idx]
        self.current_tour.append(city_idx)
        del self.remaining_nodes[remaining_node_idx]

        if self.start_node == -1:
            # this is the beginning, set the chosen city as the startpoint
            self.start_node = city_idx
            self.end_node = city_idx
        else:
            # add length of distance covered
            self.current_tour_length += self.distance(city_idx, self.start_node)
            self.start_node = city_idx

        # We reorder the remaining nodes by their distance from the start node (in ascending order)
        # remaining_nodes_distance_from_start_node = self.distance_matrix[city_idx, self.remaining_nodes]
        # sorted_idcs = np.argsort(remaining_nodes_distance_from_start_node)
        # self.remaining_nodes = [self.remaining_nodes[i] for i in sorted_idcs]

        # check if only one city is left. If so, we can automatically finish the tour.
        if len(self.remaining_nodes) == 1:
            last_node = self.remaining_nodes[0]
            self.remaining_nodes = []
            self.current_tour.append(last_node)
            self.current_tour.append(self.end_node)
            self.current_tour_length += self.distance(self.start_node, last_node)
            self.current_tour_length += self.distance(last_node, self.end_node)
            # set start node to end node.
            self.start_node = self.end_node
            self.finished = True

        return self.current_tour_length, self.finished

    def distance(self, i: int, j: int) -> float:
        """
        Returns euclidean distance between cities at idx `i` and idx `j`.
        """
        return np.sqrt(sum(np.square(self.cities[i] - self.cities[j])))

    def get_normalized_tour_length(self) -> float:
        """
        Divides the current tour length by the number of cities multiplied by \sqrt 2, which
        is the supremum of all tour lengths with n cities in the unit square.
        """
        return self.current_tour_length / (self.n_cities * math.sqrt(2))

    def get_current_situation(self):
        """
        Returns
            remaining_nodes [torch.Tensor] Tensor of shape (<num remaining nodes>, 2) consisting of the nodes which
                are currently left and from which a next node in the tour must be chosen. The remaining nodes
                are ordered (ascending) by their distance to the start node (if given)
            start_node [torch.Tensor] Tensor of shape (2). Current start node.
            end_node [torch.Tensor] Tensor of shape (2). Current end node.
            normalized_tour_length [float] Current normalized tour length.
            start_end_node_distance_matrix [np.array] Matrix of shape (2, <num remaining nodes>), where 0-th (1st) row
                corresponds to distance of start node (end node) to all remaining nodes. If start_node has not been set
                yet, `None` is returned.
            remaining_nodes_distance_matrix [np.array] Matrix of shape (<num remaining nodes>, <num remaining nodes>)
                being the distance submatrix for the remaining nodes.
        """
        # Distance matrix where the columns consist only of remaining nodes
        remaining_distance_matrix = self.distance_matrix[:, self.remaining_nodes]

        # Get the distance of start/end node to all remaining nodes.
        start_end_node_distance_matrix = remaining_distance_matrix[
            [self.start_node, self.end_node]] if self.start_node != -1 else None

        # Distance matrix with all remaining nodes in the rows
        remaining_nodes_distance_matrix = remaining_distance_matrix[self.remaining_nodes, :]

        return (
            torch.from_numpy(np.copy(self.cities[self.remaining_nodes])),
            None if self.start_node == -1 else torch.from_numpy(np.copy(self.cities[self.start_node])),
            None if self.end_node == -1 else torch.from_numpy(np.copy(self.cities[self.end_node])),
            self.get_normalized_tour_length(),
            None if start_end_node_distance_matrix is None else torch.from_numpy(
                np.copy(start_end_node_distance_matrix)),
            torch.from_numpy(np.copy(remaining_nodes_distance_matrix))
        )


class Game(BaseGame):
    """
    Class which poses the traveling salesman problem as a two-player game.
    Player 1 is referenced as "1", whereas player 2 is referenced as "-1".

    Actions are represented as integers (choosing a new destination city).
    """

    def __init__(self, instance: np.array, suppress_env_creation: bool = False, winner_in_case_of_draw: int = 1):
        """
        Initializes a new two-player game.

        Parameters
        ----------
        instance [np.array]: Cities (2-dimensional vectors) combined as an np.array in the shape (num_cities, 2). Each city
            lies in the unit square [0,1] x [0,1]
        suppress_env_creation [bool]: If `True`, no environments are created. This is only used for custom copying of games.
            See `copy`-method below.
        winner_in_case_of_draw [int]: In case of a "draw", i.e. when both players have approximately the same tour length,
            which player should win. Defaults to 1.
        """
        self.cities = instance
        self.n_cities = self.cities.shape[0]

        # keeps track of the current player on move. Player 1 always starts.
        self.current_player = 1

        self.winner_in_case_of_draw = winner_in_case_of_draw

        if not suppress_env_creation:
            self.player_environments = {
                1: TSPEnvironment(cities=copy.deepcopy(self.cities)),
                -1: TSPEnvironment(cities=copy.deepcopy(self.cities))
            }
        else:
            self.player_environments = {
                1: None,
                -1: None
            }

        # holds the players' final tour lengths
        self.player_tour_length = {
            1: float("inf"),
            -1: float("inf")
        }

        # flag indicating whether the game is finished
        self.game_is_over = False
        self.winner = 0

    def get_current_player(self):
        return self.current_player

    def get_objective(self, for_player: int) -> float:
        return self.player_tour_length[for_player]

    def get_sequence(self, for_player: int) -> float:
        return self.player_environments[for_player].current_tour

    def get_actions(self) -> List:
        """
        Legal actions for the current player is the number of remaining cities. Indices 0, ..., <num remaining cities> - 1
        will be mapped to the true cities when making a move.
        """
        environment = self.player_environments[self.current_player]
        return list(range(len(environment.remaining_nodes)))

    def is_finished_and_winner(self) -> Tuple[bool, int]:
        return self.game_is_over, self.winner

    def make_move(self, action: int):
        """
        Performs a move on the board, i.e. the current player chooses the next city from its remaining cities.

        Parameters:
            action [int]: The index of the action to play, i.e. an index in 0, ..., (number of remaining nodes - 1)

        Returns:
            game_done [bool]: Boolean indicating whether the game is over.
            reward [int]: 0, if the game is unfinished. Else "1" if the current player who made the move won, else "-1".
        """
        if self.game_is_over:
            raise Exception(f"Playing a move on a finished game!")

        current_environment: TSPEnvironment = self.player_environments[self.current_player]

        current_tour_length, finished = current_environment.add_city_to_tour(remaining_node_idx=action)

        if finished:
            self.player_tour_length[self.current_player] = current_tour_length
            # if the player is player -1, then the game is over
            # Note that we do not change player here!
            if self.current_player == -1:
                self.game_is_over = True
                self.winner = self.determine_winner()
                reward = 1 if self.current_player == self.winner else -1
                return True, reward

        # Otherwise the game is not done, and we switch player
        self.current_player *= -1
        return False, 0

    def get_current_state(self):
        """
        Returns the current game situation from the perspective of the player on move.

        Returns:
            [List[Dict]] A list with two entries, where the first entry is a dictionary
                containing the situation of the player on move, and the second entry
                the situation of the opposing player.
        """
        ret_list = []
        for player in [self.current_player, -1 * self.current_player]:
            remaining_nodes, start_node, end_node, normalized_length, \
            start_end_node_distance_matrix, remaining_nodes_distance_matrix = \
                self.player_environments[player].get_current_situation()
            is_second_player = 1 if player == -1 else 0
            ret_list.append({
                "remaining_nodes": remaining_nodes,
                "start_node": start_node,
                "end_node": end_node,
                "tour_length": normalized_length,
                "is_second_player": is_second_player,
                "start_end_node_distance_matrix": start_end_node_distance_matrix,
                "remaining_nodes_distance_matrix": remaining_nodes_distance_matrix
            })
        return ret_list

    def determine_winner(self):
        """
        Determines the winner given the tour lengths after both players finished their tours.

        If the second player's tour is not shorter than the first player's tour by at least 0.01%,
        the first player wins.

        Returns
            winner [int]: `1` for player 1 and `-1` for player 2
        """
        epsilon = 0.0001
        if self.player_tour_length[-1] < (1 - epsilon) * self.player_tour_length[1]:
            return -1
        elif math.fabs(self.player_tour_length[-1] - self.player_tour_length[1]) <= epsilon * self.player_tour_length[
            1]:
            return self.winner_in_case_of_draw
        else:
            return 1

    def copy(self):
        """
        Way faster copy than deepcopy
        """
        game = Game(instance=self.cities, suppress_env_creation=True,
                       winner_in_case_of_draw=self.winner_in_case_of_draw)
        game.current_player = self.current_player
        game.player_environments[1] = self.player_environments[1].copy()
        game.player_environments[-1] = self.player_environments[-1].copy()
        game.player_tour_length[1] = self.player_tour_length[1]
        game.player_tour_length[-1] = self.player_tour_length[-1]
        game.game_is_over = self.game_is_over
        game.winner = self.winner

        return game

    @staticmethod
    def generate_random_instance(problem_specific_config: Dict):
        return np.random.rand(problem_specific_config["N_cities"], 2)

    @staticmethod
    def random_state_augmentation(states: List[Dict]):
        return BoardGeometry.random_state_augmentation(states)


class BoardGeometry:
    """
    Helper class which augments board data by applying reflections, rotations, and scaling to
    the board data.
    """

    def __init__(self, canonical_board: List[Dict]):
        self.canonical_board = copy.deepcopy(canonical_board)

    def linear_scale(self, scale_factor: float):
        """
        Performs linear scaling of the nodes on the board.
        Parameters
        ----------
        scale_factor: [float] Maps (x, y) to (scale_factor * x, scale_factor * y)
        """
        if not (-1e-11 <= scale_factor <= 1 + 1e-11):
            print(f"WARNING: Board geometry performs linear scaling with a scale_factor of {scale_factor}. Cannot"
                  f" guarantee that all nodes lie in unit square again.")

        for situation in self.canonical_board:
            situation["remaining_nodes"] *= scale_factor
            if situation["start_node"] is not None:
                situation["start_node"] *= scale_factor
            if situation["end_node"] is not None:
                situation["end_node"] *= scale_factor
            if situation["start_end_node_distance_matrix"] is not None:
                situation["start_end_node_distance_matrix"] *= scale_factor
            situation["remaining_nodes_distance_matrix"] *= scale_factor
            situation["tour_length"] *= scale_factor

    def reflection(self, direction: int):
        """
        Performs a reflection of the board within the unit square (i.e. translation of (0.5, 0.5) to origin, reflecting
        and back)
        Parameters
        ----------
        direction: [int] 0 for horizontal reflection, 1 for vertical reflection
        Returns
        -------
        """
        if direction not in [0, 1]:
            print("WARNING: Board geometry reflection only takes 0 and 1 as direction. Not executing transformation.")
            return

        for situation in self.canonical_board:
            situation["remaining_nodes"][:, direction] = -1 * situation["remaining_nodes"][:, direction] + 1
            if situation["start_node"] is not None:
                situation["start_node"][direction] = -1 * situation["start_node"][direction] + 1
            if situation["end_node"] is not None:
                situation["end_node"][direction] = -1 * situation["end_node"][direction] + 1

    def rotate(self, angle: float):
        """
        Performs rotation around center (0.5, 0.5) of unit square by `angle` given in radians.
        As the rotation does not guarantee that all nodes lie again in the unit square, afterwards a
        linear scaling is performed if necessary.
        Parameters
        ----------
        angle: [float] Rotation angle in radions
        """
        max_coordinate = -1  # keeps track of maximum absolute coordinate to apply scaling if necessary
        cos = np.cos(angle)
        sin = np.sin(angle)

        self.translate(-0.5, -0.5)

        for situation in self.canonical_board:
            new_x = cos * situation["remaining_nodes"][:, 0] - sin * situation["remaining_nodes"][:, 1]
            new_y = sin * situation["remaining_nodes"][:, 0] + cos * situation["remaining_nodes"][:, 1]
            situation["remaining_nodes"][:, 0] = new_x
            situation["remaining_nodes"][:, 1] = new_y

            if situation["start_node"] is not None:
                new_x = cos * situation["start_node"][0] - sin * situation["start_node"][1]
                new_y = sin * situation["start_node"][0] + cos * situation["start_node"][1]
                situation["start_node"][0] = new_x
                situation["start_node"][1] = new_y
            if situation["end_node"] is not None:
                new_x = cos * situation["end_node"][0] - sin * situation["end_node"][1]
                new_y = sin * situation["end_node"][0] + cos * situation["end_node"][1]
                situation["end_node"][0] = new_x
                situation["end_node"][1] = new_y

            # We have translated the board to the origin and rotated. Might be that now the absolute value
            # of a coordinate is greater than 0.5, which means that if we translate back we will no longer be in
            # unit square
            max_coordinate_in_situation = max(
                -1 if situation["remaining_nodes"].shape[0] == 0 else float(
                    torch.max(torch.abs(situation["remaining_nodes"]))),
                -1 if situation["start_node"] is None else float(torch.max(torch.abs(situation["start_node"]))),
                -1 if situation["end_node"] is None else float(torch.max(torch.abs(situation["end_node"])))
            )
            if max_coordinate_in_situation > max_coordinate:
                max_coordinate = max_coordinate_in_situation

        if max_coordinate > 0.5:
            self.linear_scale(scale_factor=0.5 / max_coordinate)

        self.translate(0.5, 0.5)

    def translate(self, x: float, y: float):
        """
        Adds vector (x, y) to each node.
        """
        for situation in self.canonical_board:
            situation["remaining_nodes"][:, 0] += x
            situation["remaining_nodes"][:, 1] += y
            if situation["start_node"] is not None:
                situation["start_node"][0] += x
                situation["start_node"][1] += y
            if situation["end_node"] is not None:
                situation["end_node"][0] += x
                situation["end_node"][1] += y

    def get_board(self):
        return self.canonical_board

    @staticmethod
    def random_state_augmentation(states: List[Dict]):
        """
        Performs a random reflection, scaling and rotation to the board.

        Parameters
            canonical_board: Canonical board as obtained from `TSPGame.get_current_canonical_board`

        Returns
            augmented canonical_board
        """
        board_geometry = BoardGeometry(states)

        # Apply random reflection
        if random.randint(0, 1) == 0:
            board_geometry.reflection(direction=random.randint(0, 1))

        # Apply random rotation
        if random.randint(0, 1) == 0:
            angle = np.random.random() * 2 * np.pi
            board_geometry.rotate(angle)

        # Apply random scaling
        if random.randint(0, 1) == 0:
            scale_factor = np.random.random()
            board_geometry.linear_scale(scale_factor)

        return board_geometry.get_board()
