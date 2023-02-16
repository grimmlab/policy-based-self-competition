import math
import torch
import numpy as np
import copy
import random
from typing import Tuple, List, Dict, Optional
from base_game import BaseGame


class JSSPEnvironment:
    """
    Environment for one player, representing the current state of the scheduling problem.
    """
    def __init__(self, instance: np.array, suppress_operation_mapping: bool = False):
        self.instance = instance  # problem instance of shape (J, O, M)
        self.num_jobs = instance.shape[0]
        self.num_operations = instance.shape[1]
        self.num_machines = instance.shape[2]

        self.job_schedule_sequence = []  # holds the sequence of scheduled job idcs
        # holds for each job the last operation idx which has been scheduled so far
        self.job_last_op_scheduled = [-1] * self.num_jobs
        # Holds for each job the earliest time the next operation can start (as
        # an earlier operation of the job might block on some machine)
        self.job_availability_times = [0.] * self.num_jobs
        # Holds for each machine the time when they can be next accessed
        self.machine_availability_times = [0.] * self.num_machines
        # Keeps track of which jobs are "remaining", i.e. for which jobs there are still operations
        # which need to be scheduled. This will be used to map indices 0, ..., <num remaining jobs> to
        # the true job indices of the original problem instance.
        self.remaining_job_idcs: List = list(range(self.num_jobs))

        if not suppress_operation_mapping:
            # Operation to machine index mapping for faster lookup
            self.operation_to_machine_idx = np.zeros((self.num_jobs, self.num_operations), dtype=np.int32)
            # Operation to processing time mapping for faster lookup
            self.operation_to_processing_time = np.zeros((self.num_jobs, self.num_operations), dtype=np.float32)

            for job_idx in range(self.num_jobs):
                for op_idx in range(self.num_operations):
                    machine_idx = np.nonzero(self.instance[job_idx, op_idx])[0][0]
                    processing_time = self.instance[job_idx, op_idx, machine_idx]
                    self.operation_to_machine_idx[job_idx, op_idx] = int(machine_idx)
                    self.operation_to_processing_time[job_idx, op_idx] = processing_time

        self.finished = False

    def add_job_idx_to_schedule(self, job_idx: int):
        """
        Adds the next operation of a job by its index (index in original instance!) to the schedule.

        Parameters
        ----------
        job_idx [int] Index of job of which the next operation should be scheduled.

        Returns
        -------

        """
        # operation of job to schedule
        if self.job_last_op_scheduled[job_idx] == self.num_operations - 1:
            raise Exception(f"Trying to schedule an already finished job. Job idx: {job_idx}")

        self.job_schedule_sequence.append(job_idx)
        operation_idx = self.job_last_op_scheduled[job_idx] + 1
        self.job_last_op_scheduled[job_idx] = operation_idx

        # get the machine and processing time of the operation
        machine_idx = self.operation_to_machine_idx[job_idx, operation_idx]
        processing_time = self.operation_to_processing_time[job_idx, operation_idx]
        # a job can only be started when its last operation is finished and the machine is free
        start_time = max(self.machine_availability_times[machine_idx], self.job_availability_times[job_idx])
        end_time = start_time + processing_time

        # update availability times
        self.job_availability_times[job_idx] = end_time
        self.machine_availability_times[machine_idx] = end_time

        if operation_idx == self.num_operations - 1:
            # all operations of job have been scheduled. remove its index from remaining_job_idcs
            self.remaining_job_idcs.remove(job_idx)

        if not len(self.remaining_job_idcs):
            self.finished = True
        # autofinish if there is only one operation left to schedule
        elif len(self.remaining_job_idcs) == 1 and self.job_last_op_scheduled[self.remaining_job_idcs[0]] == self.num_operations - 2:
            self.add_job_idx_to_schedule(self.remaining_job_idcs[0])

        return self.get_current_maketime(), self.finished

    def add_remaining_job_idx_to_schedule(self, remaining_job_idx: int):
        job_idx = self.remaining_job_idcs[remaining_job_idx]
        return self.add_job_idx_to_schedule(job_idx)

    def get_current_maketime(self):
        return max(self.machine_availability_times)

    def get_current_continuous_situation(self):
        """
        Returns
        -------
        Current state of the job schedule environment as a tuple, consisting of:

        operation_machine_idcs: [torch.LongTensor] of shape (num remaining jobs, num operations),
            where (j, i)-th entry is the machine index of operation i of job j
        operation_processing_times: [torch.Tensor] of shape (num remaining jobs, num operations, 1),
            where (j, i, 0)-th entry is the processing time of operation i of job j
        num_remaining_jobs: [int] Number of remaining jobs
        job_availability_tensor: [torch.FloatTensor] Relative job availability for remaining jobs.
        min_machine_time: [float] Minimum absolute machine time, from which the situation is taken relatively
        machine_times: [torch.FloatTensor] of length (num machines) with relative machine times.
        num_ops_per_job: [List[int]] containing number of remaining operations for each remaining job
        """
        min_machine_time = min(self.machine_availability_times)
        num_ops_per_job = []  # holds number of remaining operations for each remaining job
        num_remaining_jobs = len(self.remaining_job_idcs)

        # get the availability times of machines and jobs. We "normalize" the situation by shifting the times of
        # machines and jobs to the left by the minimum machine availability time.
        machine_availability_tensor = torch.FloatTensor(self.machine_availability_times)
        machine_availability_tensor -= min_machine_time
        job_availability_tensor = torch.FloatTensor([self.job_availability_times[j] for j in self.remaining_job_idcs])
        job_availability_tensor -= min_machine_time
        job_availability_tensor[job_availability_tensor < 0] = 0.

        operation_machine_idcs = torch.zeros((len(self.remaining_job_idcs), self.num_operations), dtype=torch.long)
        operation_processing_times = np.zeros((len(self.remaining_job_idcs), self.num_operations), dtype=np.float32)
        for j, job_idx in enumerate(self.remaining_job_idcs):
            ops_start_idx = self.job_last_op_scheduled[job_idx] + 1  # index of first operation which can be scheduled
            num_remaining_operations = self.num_operations - ops_start_idx  # how many operations are left for this job
            num_ops_per_job.append(num_remaining_operations)
            for i in range(ops_start_idx, self.num_operations):
                operation_machine_idcs[j, i - ops_start_idx] = self.operation_to_machine_idx[job_idx, i]
                operation_processing_times[j, i - ops_start_idx] = self.operation_to_processing_time[job_idx, i]

        return (operation_machine_idcs,
                torch.from_numpy(operation_processing_times),
                num_remaining_jobs,
                job_availability_tensor,
                min_machine_time,
                machine_availability_tensor,
                num_ops_per_job
                )

    def copy(self):
        env = JSSPEnvironment(self.instance, suppress_operation_mapping=True)

        env.job_schedule_sequence = self.job_schedule_sequence.copy()
        env.job_last_op_scheduled = self.job_last_op_scheduled.copy()
        env.job_availability_times = self.job_availability_times.copy()
        env.machine_availability_times = self.machine_availability_times.copy()
        env.remaining_job_idcs = self.remaining_job_idcs.copy()
        env.operation_to_processing_time = self.operation_to_processing_time
        env.operation_to_machine_idx = self.operation_to_machine_idx
        env.finished = self.finished

        return env


class Game(BaseGame):
    """
    Class which poses the job shop scheduling problem as a two-player game.
    Player 1 is referenced as "1", whereas player 2 is referenced as "-1".

    Actions are represented as integers (choosing a remaining job to schedule the next operation from).
    """
    def __init__(self, instance: np.array, suppress_env_creation: bool = False, winner_in_case_of_draw: int = 1):
        """
        Initializes a new two-player game.

        Parameters
        ----------
        instance [np.array]: JSP Problem instance
        suppress_env_creation [bool]: If `True`, no environments are created. This is only used for custom copying of games.
            See `copy`-method below.
        winner_in_case_of_draw [int]: In case of a "draw", i.e. when both players have approximately the same tour length,
            which player should win. Defaults to 1.
        """
        self.instance = instance

        # keeps track of the current player on move. Player 1 always starts.
        self.current_player = 1

        self.winner_in_case_of_draw = winner_in_case_of_draw

        if not suppress_env_creation:
            self.player_environments = {
                1: JSSPEnvironment(instance=self.instance),
                -1: JSSPEnvironment(instance=self.instance)
            }
        else:
            self.player_environments = {
                1: None,
                -1: None
            }

        # holds the players' final schedule maketimes
        self.player_maketimes = {
            1: float("inf"),
            -1: float("inf")
        }

        # flag indicating whether the game is finished
        self.game_is_over = False
        self.winner = 0

    def get_current_player(self):
        return self.current_player

    def get_objective(self, for_player: int) -> float:
        return self.player_maketimes[for_player]

    def get_sequence(self, for_player: int) -> List[int]:
        return self.player_environments[for_player].job_schedule_sequence

    def get_actions(self) -> List:
        """
        Legal actions for the current player is the number of remaining jobs. Indices 0, ..., <num remaining job> - 1
        will be mapped to the true jobs when making a move.
        """
        environment = self.player_environments[self.current_player]
        return list(range(len(environment.remaining_job_idcs)))

    def is_finished_and_winner(self) -> Tuple[bool, int]:
        return self.game_is_over, self.winner

    def make_move(self, action: int):
        """
        Performs a move on the board, i.e. the current player chooses the next job from its remaining jobs.

        Parameters:
            action [int]: The index of the action to play, i.e. an index in 0, ..., (number of remaining jobs - 1)

        Returns:
            game_done [bool]: Boolean indicating whether the game is over.
            reward [int]: 0, if the game is unfinished. Else "1" if the current player who made the move won, else "-1".
        """
        if self.game_is_over:
            raise Exception(f"Playing a move on a finished game!")

        current_environment: JSSPEnvironment = self.player_environments[self.current_player]

        current_maketime, finished = current_environment.add_remaining_job_idx_to_schedule(remaining_job_idx=action)

        if finished:
            self.player_maketimes[self.current_player] = current_maketime
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
            (
                operation_machine_idcs,
                operation_processing_times,
                num_remaining_jobs,
                job_availability_tensor,
                min_machine_time,
                machine_availability_tensor,
                num_ops_per_job
             ) = self.player_environments[player].get_current_continuous_situation()
            is_second_player = 1 if player == -1 else 0
            ret_list.append({
                "operation_machine_idcs": operation_machine_idcs,
                "operation_processing_times": operation_processing_times,
                "num_remaining_jobs": num_remaining_jobs,
                "job_availability_tensor": job_availability_tensor,
                "machine_availability_tensor": machine_availability_tensor,
                "min_machine_time": min_machine_time,
                "num_ops_per_job": num_ops_per_job,
                "is_second_player": is_second_player
            })

        return ret_list

    def determine_winner(self):
        """
        Determines the winner given the maketimes after both players finished their schedules.

        If the second player's maketime is not shorter than the first player's tour by at least 0.01%,
        the first player wins.

        Returns
            winner [int]: `1` for player 1 and `-1` for player 2
        """
        epsilon = 0.0001
        if self.player_maketimes[-1] < (1 - epsilon) * self.player_maketimes[1]:
            return -1
        elif math.fabs(self.player_maketimes[-1] - self.player_maketimes[1]) <= epsilon * self.player_maketimes[
            1]:
            return self.winner_in_case_of_draw
        else:
            return 1

    def copy(self):
        """
        Way faster copy than deepcopy
        """
        game = Game(instance=self.instance, suppress_env_creation=True,
                       winner_in_case_of_draw=self.winner_in_case_of_draw)
        game.current_player = self.current_player
        game.player_environments[1] = self.player_environments[1].copy()
        game.player_environments[-1] = self.player_environments[-1].copy()
        game.player_maketimes[1] = self.player_maketimes[1]
        game.player_maketimes[-1] = self.player_maketimes[-1]
        game.game_is_over = self.game_is_over
        game.winner = self.winner

        return game

    @staticmethod
    def generate_random_instance(problem_specific_config: Dict):
        return JSSPInstanceGenerator.random_instance(problem_specific_config["num_jobs"],
                                                     problem_specific_config["num_machines"])

    @staticmethod
    def random_state_augmentation(states: List[Dict]):
        return BoardGeometry.random_state_augmentation(states)


class BoardGeometry:
    """
    Helper class which augments board data by applying machine permutations, linear scaling of maketimes.
    """
    def __init__(self, canonical_board: List[Dict]):
        self.canonical_board = copy.deepcopy(canonical_board)
        self.num_machines = len(self.canonical_board[0]["machine_availability_tensor"])

    def linear_scale(self, scale_factor: float):
        """
        Performs linear scaling of the nodes on the board.
        Parameters
        ----------
        scale_factor: [float] Maps (x, y) to (scale_factor * x, scale_factor * y)
        """
        if not (-1e-11 <= scale_factor <= 1 + 1e-11):
            print(f"WARNING: Board geometry performs linear scaling with a scale_factor of {scale_factor}. Cannot"
                  f" guarantee that all maketimes lie in [0, 1] again.")

        for situation in self.canonical_board:
            situation["operation_processing_times"] *= scale_factor
            situation["job_availability_tensor"] *= scale_factor
            situation["machine_availability_tensor"] *= scale_factor
            situation["min_machine_time"] *= scale_factor
            if "baseline_outcome" in situation:
                situation["baseline_outcome"] *= scale_factor

    def switch_machines(self, machine_idx_1: int, machine_idx_2: int):
        assert machine_idx_1 < self.num_machines and machine_idx_2 < self.num_machines

        for situation in self.canonical_board:
            situation["operation_machine_idcs"][situation["operation_machine_idcs"] == machine_idx_1] = -1
            situation["operation_machine_idcs"][situation["operation_machine_idcs"] == machine_idx_2] = machine_idx_1
            situation["operation_machine_idcs"][situation["operation_machine_idcs"] == -1] = machine_idx_2

            # switch availability times
            avail_1 = situation["machine_availability_tensor"][machine_idx_1].item()
            avail_2 = situation["machine_availability_tensor"][machine_idx_2].item()
            situation["machine_availability_tensor"][machine_idx_1] = avail_2
            situation["machine_availability_tensor"][machine_idx_2] = avail_1

    def get_board(self):
        return self.canonical_board

    @staticmethod
    def random_state_augmentation(states: List[Dict]):
        board_geometry = BoardGeometry(states)

        # Apply multiple random machine switches
        for _ in range(10):
            idx_1 = random.randint(0, board_geometry.num_machines - 1)
            idx_2 = random.randint(0, board_geometry.num_machines - 1)
            board_geometry.switch_machines(idx_1, idx_2)

        # Apply random scaling
        #if random.randint(0, 1) == 0:
        #    scale_factor = np.random.random()
        #    board_geometry.linear_scale(scale_factor)

        return board_geometry.get_board()


class JSSPInstanceGenerator:
    """
    Helper class to geenrate job shop scheduling instances as described by Taillard in "Benchmarks for basic scheduling
    problems".
    """
    @staticmethod
    def random_instance_taillard(num_jobs: int, num_machines: int) -> np.array:
        """
        Docstring uses Taillard notation.

        Returns
        -------
            [np.array] of shape (J, M, M), where entry (j, i, k) = 0 if k != M_ij else d_ij.
        """
        num_operations = num_machines  # num operations for each job equal number of machines for Taillard.

        # d_ij matrix of processing times. ij entry is processing time of j-th operation of i-th job, which is
        # an integer between 1 and 99.
        processing_time_matrix = np.random.uniform(low=0, high=1, size=(num_jobs, num_operations))

        instance = np.zeros((num_jobs, num_operations, num_machines))

        # Step 0 in Taillard (p. 281-282):
        for j in range(num_jobs):
            for i in range(num_operations):
                instance[j, i, i] = processing_time_matrix[j, i]

        # Step 1 in Taillard:
        for j in range(num_jobs):
            for i in range(num_operations - 1):
                random_op = random.randint(i + 1, num_machines - 1)
                M_ij = np.nonzero(instance[j, i])[0][0]
                M_Uj = np.nonzero(instance[j, random_op])[0][0]
                _temp = instance[j, i, M_ij]
                instance[j, i, M_Uj] = instance[j, i, M_ij]
                instance[j, i, M_ij] = 0
                instance[j, random_op, M_ij] = instance[j, random_op, M_Uj]
                instance[j, random_op, M_Uj] = 0

        return instance

    @staticmethod
    def random_instance(num_jobs: int, num_machines: int) -> np.array:
        """
        Docstring uses Taillard notation.

        Returns
        -------
            [np.array] of shape (J, M, M), where entry (j, i, k) = 0 if k != M_ij else d_ij.
        """
        num_operations = num_machines  # num operations for each job equal number of machines for Taillard.

        # d_ij matrix of processing times. ij entry is processing time of j-th operation of i-th job, which is
        # an integer between 1 and 99.
        processing_time_matrix = np.random.uniform(low=0, high=1, size=(num_jobs, num_operations))

        instance = np.zeros((num_jobs, num_operations, num_machines))

        for j in range(num_jobs):
            machines = np.random.permutation(num_machines)
            for i in range(num_operations):
                instance[j, i, machines[i]] = processing_time_matrix[j, i]

        return instance

    @staticmethod
    def read_taillard_instance(file_path: str) -> np.array:
        with open(file_path, "r") as f:
            lines = f.readlines()

        # first line is number of jobs and number of machines
        first = lines[0].split()
        num_jobs = int(first[0])
        num_machines = int(first[1])
        num_operations = num_machines

        # Setup empty instance
        instance = np.zeros((num_jobs, num_operations, num_machines))

        for j in range(num_jobs):
            line = lines[j + 1].split()

            for i in range(num_operations):
                # we always have pairs in the line, where first entry is the machine, and the second entry is the
                # processing time. Machines are indexed from 0
                machine = int(line[i*2])
                time = int(line[i*2 + 1])
                instance[j, i, machine] = time

        return instance

    @staticmethod
    def read_taillard_solution_and_convert_to_job_sequence(file_path_instance: str, file_path_solution: str) -> Tuple[int, List[int]]:
        """
        Reads in a solution to a Taillard problem and converts it into a single sequence of jobs.

        Quote: "In the solutions row i,column k gives the start time
        of job i on machine k."

        Returns:
            [int] Full makespan of solution
            [List[int]] Sequence of job indices resulting in this solution
        """
        instance = JSSPInstanceGenerator.read_taillard_instance(file_path_instance)
        num_jobs = instance.shape[0]
        num_machines = num_operations = instance.shape[2]

        with open(file_path_solution, "r") as f:
            lines = f.readlines()

        makespan = int(lines[0])

        machine_schedules = [[] for _ in range(num_machines)]

        for j in range(num_jobs):
            machine_start_times = lines[j + 1].split()
            for m in range(num_machines):
                i = np.nonzero(instance[j, :, m])[0][0]  # operation index

                start_time = int(machine_start_times[m])
                machine_schedule = machine_schedules[m]
                is_scheduled = False
                for k, (_, _, t) in enumerate(machine_schedule):
                    if start_time < t:
                        is_scheduled = True
                        # prepend current operation
                        machine_schedules[m] = machine_schedule[:k] + [(j, i, start_time)] + machine_schedule[k:]
                        break
                if not is_scheduled:
                    machine_schedule.append((j, i, start_time))

        # From the machine schedules construct a single job sequence now
        job_sequence = []

        job_ops_scheduled = [-1] * num_jobs

        while sum([len(schedule) for schedule in machine_schedules]) > 0:
            for m in range(num_machines):
                schedule = machine_schedules[m]
                if len(schedule):
                    j, i, _ = schedule[0]
                    if job_ops_scheduled[j] == i-1:
                        job_sequence.append(j)
                        job_ops_scheduled[j] = i
                        machine_schedules[m] = schedule[1:]

        return makespan, job_sequence