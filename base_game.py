from abc import ABC, abstractmethod

from typing import Tuple, List, Dict


class BaseGame(ABC):
    """
    Class which represents the base environment for GAZ planning problems.
    """
    @abstractmethod
    def get_actions(self) -> List[int]:
        """
        Legal actions for the current player given as a list of ints.
        """
        pass

    @abstractmethod
    def make_move(self, action: int) -> Tuple[bool, float]:
        """
        Performs a move in the environment

        Parameters:
            action [int]: The index of the action to play

        Returns:
            episode_done [bool]: Boolean indicating whether the episode is over.
            reward [float]: Reward resulting from move
        """
        pass

    @abstractmethod
    def get_current_state(self):
        """
        Returns the current situation.
        """
        pass

    @abstractmethod
    def get_current_player(self):
        """
        Returns an int (1 or -1) indicating the player which is to move.
        For single-player setups this can just return 1 all of the time.
        """
        pass

    @abstractmethod
    def get_objective(self, for_player: int) -> float:
        """
        Returns the objective of the current problem for a given player
        """
        pass

    @abstractmethod
    def get_sequence(self, for_player: int) -> List[int]:
        """
        Returns action sequence for given player
        """
        pass

    @abstractmethod
    def is_finished_and_winner(self) -> Tuple[bool, int]:
        pass

    @abstractmethod
    def copy(self):
        """
        Individual method to speed up copying in MCTS.
        """
        pass

    @staticmethod
    @abstractmethod
    def generate_random_instance(problem_specific_config: Dict):
        """
        Generate a random instance given the problem specific config dictionary.
        """
        pass

    @staticmethod
    @abstractmethod
    def random_state_augmentation(states: List[Dict]):
        """
        Performs some kind of random state augmentation (if applicable, else return the states)
        Parameters
            states: List of states as obtained from `BaseGame.get_current_state`
        Returns
            List[Dict]: augmented states of the same format
        """