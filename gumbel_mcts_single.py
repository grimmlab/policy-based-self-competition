import threading

import ray
import torch
import time
import numpy as np

from typing import Optional, List, Dict, Tuple

from base_game import BaseGame
from base_config import BaseConfig
from inferencer import ModelInferencer
from gumbel_mcts import GumbelMCTS


class SingleplayerGumbelNode:
    """
    Represents one node in the search tree, i.e. the representation of a
    certain state, stemming from a parent state coupled with an action.
    """
    def __init__(self, prior: float, prior_logit: float, parent_node, parent_action: int, timestep: int = 0):
        """
        Parameters
        ----------
        prior [float]: Prior probability of selecting this node.
        prior_logit [flaot]: Logit corresponding to prior.
        parent_node [GumbelNode]: Parent node from which state coupled with action this node
            results.
        parent_action [int]: Action that was taken in parent node which led to this node.
        """
        self.visit_count = 0
        self.prior = prior
        self.prior_logit = prior_logit
        self.value_sum = 0  # sum of backpropagated estimated values. Corresponds to "W(s, a)" in the Alpha Zero paper
        self.children = {}  # mapping of action -> child node
        self.reward = 0  # reward obtained when applying action to parent
        self.state: Optional[BaseGame] = None  # holds the state of the node as an instance copy of the game

        self.predicted_value = 0  # The value predicted by the network (resp. true value if game is terminal in this node)

        self.expanded = False
        self.terminal = False  # whether in this node the game is finished
        self.timestep = timestep  # keeps track of the timestep of the state this node corresponds to
        self.value_baseline: Optional[float] = None  # An optional value baseline which is used in this node instead
                                                     # of the mixed value estimation

        # keeps a torch.Tensor of the predicted logits for child actions for easier access
        self.children_prior_logits: Optional[torch.Tensor] = None

        # keeps track of the node's parent and which action led to this one
        self.parent_node = parent_node
        self.parent_action = parent_action

        # If the node is root at some point, holds the final chosen action after sequential halving
        self.sequential_halving_chosen_action: int = -1

    def value(self) -> float:
        """
        Returns the state-action value of this node (i.e. Q(s, a) where `s` is parent state
        and `a` is action leading to this node, depending on the visit counts (corresponds to Q(s,a) in the Alpha
        Zero paper) from perspective of player on move.
        """
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, policy_logits: Optional[torch.Tensor], predicted_value: Optional[float], reward: float, state: BaseGame,
               value_baseline: Optional[float] = None) -> bool:
        """
        Expands the node by branching out actions.

        Parameters:
            policy_logits [torch.Tensor]: Probability logits for all actions as 1-dim torch tensor.
            predicted_value [float]: Value predicted in this state by network (or true value if it is a terminal state)
            state [TSP]: Game state in this node.
        Returns:
            Success [bool]: Will be `False`, if this is a dead end and there are no legal actions.
        """
        if self.expanded:
            raise Exception("Node already expanded")
        self.expanded = True
        self.state = state
        self.predicted_value = predicted_value
        self.reward = reward

        if self.state.game_is_over:
            self.terminal = True
            # this is a terminal state, so there are no future rewards
            self.predicted_value = 0
            return True

        if value_baseline is not None:
            self.value_baseline = value_baseline

        self.children_prior_logits = policy_logits
        policy_probs = torch.softmax(policy_logits, dim=0).numpy().astype('float64')
        # normalize, in most cases the sum is not exactly equal to 1 which is problematic when sampling
        policy_probs /= policy_probs.sum()

        for action, p in enumerate(policy_probs):
            self.children[action] = SingleplayerGumbelNode(prior=p, prior_logit=policy_logits[action],
                                               parent_node=self, parent_action=action, timestep=self.timestep + 1)
        return True

    def get_visit_count_distribution_tensor(self) -> torch.Tensor:
        visit_counts = torch.tensor([1. * child.visit_count for child in self.children.values()])
        return visit_counts / (1. + visit_counts.sum())

    def get_estimated_q_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the estimated Q-values for the actions taken in this node.
        IMPORTANT! Note that the q-value for each action must always be from the perspective
        of the player in this node, so we might need to change the sign of the values.
        """
        values = []
        is_unvisited = []
        for child in self.children.values():
            values.append(child.value())
            is_unvisited.append(True if child.visit_count == 0 else False)
        return torch.tensor(values), torch.tensor(is_unvisited)

    def get_completed_q_values(self) -> torch.Tensor:
        completed_q, unvisited_children = self.get_estimated_q_tensor()
        value_approximation = self.get_mixed_value_approximation() if self.value_baseline is None else self.value_baseline
        # we assume that if the value of a child is exactly 0, then it has not been visited. This is not entirely
        # correct, as the values in the trajectories might cancel out, however is very unlikely.
        completed_q[unvisited_children] = value_approximation

        return completed_q

    def get_mixed_value_approximation(self) -> float:
        sum_visits = sum([child.visit_count for child in self.children.values()])
        sum_visited_pi = 0
        sum_visited_pi_q = 0
        for action in self.children:
            node = self.children[action]
            if node.visit_count > 0:
                pi = (1. * node.visit_count) / sum_visits
                sum_visited_pi += pi
                sum_visited_pi_q += pi * node.value()

        mixed_value = self.predicted_value
        if sum_visited_pi != 0:
            mixed_value += (sum_visits / sum_visited_pi) * sum_visited_pi_q
        mixed_value /= 1. + sum_visits

        return mixed_value


class SingleplayerGumbelMCTS:
    """
    Core Monte Carlo Tree Search using Planning with Gumbel.
    The tree persists over the full game. We run N simulations using sequential halving at the root,
    and traverse the tree according to the "non-root" selection formula in Planning with Gumbel,
    which gets then expanded.
    """
    def __init__(self, actor_id: int, config: BaseConfig, model_inference_worker: ModelInferencer,
                 deterministic: bool, min_max_normalization: bool, model_to_use: str = "newcomer",
                 value_estimates: Optional[np.array] = None):
        """
        Parameters
            actor_id [int] Unique Identifier which is used to mark inference queries as belonging to this tree.
            config [BaseConfig]
            model_inference_worker: Inference worker
            deterministic [bool]: If True, sampled Gumbels are set to zero at root, i.e. actions with
                maximum predicted logits are selected (no sampling)
            min_max_normalization [bool]: If True, Q-values are normalized by min-max values in tree,
                as in (Gumbel) MuZero.
            model_to_use [str]: "newcomer" or "best", depending on which model should be used for inference.
                For Single Vanilla/N-Step this should be set to "newcomer", and "best" for greedy rollouts
                in Greedy Scalar.
        """
        self.id = actor_id
        self.config = config
        self.inference_worker = model_inference_worker
        self.deterministic = deterministic
        self.model_to_use = model_to_use
        self.min_max_normalization = min_max_normalization
        self.value_estimates: Optional[np.array] = value_estimates

        self.root: SingleplayerGumbelNode = SingleplayerGumbelNode(prior=0, prior_logit=0, parent_node=None, parent_action=-1)

        # incrementing counter for simulations queried in this tree; used to identify returned queries
        self.query_counter = 0
        # Stores tuples of (actor_id, query_id, tensor, model_name) which need to be
        # sent to the model inference process.
        # In order not to send the leaf node queries for each simulation individually, we batch them
        # in each sequential halving level so to have only one roundtrip.
        self.model_keys = [self.model_to_use]

        self.query_states = dict()
        self.query_ids = dict()
        for key in self.model_keys:
            self.query_states[key] = []
            self.query_ids[key] = []
        self.queries_num_actions = dict()
        self.query_results_lock: threading.Lock = threading.Lock()
        self.query_results = dict()

        # Track the maximum search depth in the tree for a move
        self.search_depth = 0

        # Keeps track of full number of actions for a move when running at root
        self.num_actions_for_move = 0

        self.waiting_time = 0

        self.min_max_stats = MinMaxStats()  # for normalizing Q values

    def add_state_to_prediction_queue(self, game_state: BaseGame) -> str:
        """
        Adds state to queue which is prepared for inference worker, and returns a
        query id which can be used to poll the results.
        """
        state = game_state.get_current_state()
        num_actions = len(game_state.get_actions())

        self.query_counter += 1
        query_id = str(self.query_counter)

        # unpack the canonical board so that the numpy arrays are not copied to the inferencer
        self.query_states[self.model_to_use].append(state)
        self.query_ids[self.model_to_use].append(query_id)
        self.queries_num_actions[query_id] = num_actions

        return query_id

    def dispatch_prediction_queue(self):
        """
        Sends the current inferencing queue to the inference worker, if
        the queue is not empty. Empties the query list afterwards.
        """
        if len(self.query_ids[self.model_to_use]):
            if not self.config.inference_on_experience_workers:
                ray.get(self.inference_worker.add_list_to_queue.remote(self.id, self.query_ids,
                                                                       self.query_states, list(self.query_states.keys())))
            else:
                query_ids = self.query_ids[self.model_to_use]
                policy_logits_batch, value_batch = self.inference_worker.infer_batch(self.query_states[self.model_to_use],
                                                                                     self.model_to_use)
                for i, query_id in enumerate(query_ids):
                    policy_logits = torch.from_numpy(policy_logits_batch[i].copy())
                    value = value_batch[i][0]
                    self.query_results[query_id] = (policy_logits, value)

            self.query_ids[self.model_to_use] = []
            self.query_states[self.model_to_use] = []

    def add_query_results(self, results):
        with self.query_results_lock:
            query_ids, policy_logits_batch, value_batch = results
            for i, query_id in enumerate(query_ids):
                policy_logits = torch.from_numpy(policy_logits_batch[i].copy())
                value = value_batch[i][0]

                self.query_results[query_id] = (policy_logits, value)

    def check_for_prediction_result(self, query_id: str) -> Optional[Tuple]:
        result = None
        if query_id in self.query_results:
            with self.query_results_lock:
                policy_logits_padded, value = self.query_results[query_id]
                policy_logits = policy_logits_padded[:self.queries_num_actions[query_id]]
                result = (policy_logits, value)
                del self.query_results[query_id]
                del self.queries_num_actions[query_id]
        return result

    def wait_for_prediction_results(self, query_ids: List[str]) -> Dict:
        results = dict()
        waiting_time = time.perf_counter()
        for query_id in query_ids:
            res = None
            while res is None:
                time.sleep(0)
                res = self.check_for_prediction_result(query_id)
            results[query_id] = res
        self.waiting_time += time.perf_counter() - waiting_time
        return results

    def expand_root(self, game: BaseGame):
        state = game.copy()
        query_id = self.add_state_to_prediction_queue(state)
        self.dispatch_prediction_queue()

        while not query_id in self.query_results:
            time.sleep(0)

        policy_logits, value = self.check_for_prediction_result(query_id)
        reward = 0  # reward is irrelevant for root as we are only interested in what happens from here on
        value_baseline = None if self.value_estimates is None else self.value_estimates[self.root.timestep]
        self.root.expand(policy_logits, value, reward, state, value_baseline)

    def run_at_root(self, game: BaseGame, n_actions: int, only_expand_root: bool = False) -> Tuple[SingleplayerGumbelNode, Dict]:
        self.num_actions_for_move = n_actions

        # Step 1: If the root is not expanded, we expand it
        if not self.root.expanded:
            self.expand_root(game)

        # Step 2: Check if the current player should simulate moves for policy improvement.
        # If not, we return right after expanding the root
        if n_actions == 1 or only_expand_root:
            return self.root, {}

        # Step 3: Sample `n_actions` Gumbel variables for sampling without replacement.
        if self.deterministic:
            # No gumbel sampling, use pure logits.
            gumbel_logits = np.zeros(n_actions)
        else:
            gumbel_logits = np.random.gumbel(size=n_actions)
        gumbel_logits_tensor = torch.from_numpy(gumbel_logits)
        gumbel_logits_tensor += self.root.children_prior_logits
        gumbel_logits = gumbel_logits_tensor.numpy()

        # Step 4: Using the Gumbel variables, do the k-max trick to sample actions.
        n_actions_to_sample = min(n_actions, self.config.gumbel_sample_n_actions, self.config.num_simulations)
        if n_actions_to_sample == n_actions:
            # no sampling needed, we consider all remaining actions
            considered_actions = list(range(n_actions))
        else:
            # get the indices of the top k gumbel logits
            considered_actions = list(np.argpartition(gumbel_logits, -n_actions_to_sample)[-n_actions_to_sample:])
            considered_actions.sort()

        # Step 5: We now need to check how many simulations we may use in each level of
        # sequential halving.
        num_actions_per_level, num_simulations_per_action_and_level = \
            GumbelMCTS.get_sequential_halving_simulations_for_levels(n_actions_to_sample, self.config.num_simulations)

        # Step 6: Perform sequential halving and successively eliminate actions
        for level, num_simulations_per_action in enumerate(num_simulations_per_action_and_level):
            self.run_simulations_for_considered_root_actions(
                considered_actions=considered_actions,
                num_simulations_per_action=num_simulations_per_action
            )

            # get the sigma-values of the estimated q-values at the root after the simulations
            # for this level
            estimated_q_tensor, _ = self.root.get_estimated_q_tensor()
            if self.min_max_normalization:
                self.min_max_stats.update(estimated_q_tensor)
                estimated_q_tensor = self.min_max_stats.normalize(estimated_q_tensor)
            updated_gumbels = gumbel_logits_tensor + self.sigma_q(self.root, estimated_q_tensor)
            considered_gumbels = updated_gumbels[considered_actions]

            if level < len(num_simulations_per_action_and_level) - 1:
                # choose the maximum k number of gumbels, where k is the number of actions for
                # next level. Note that we have to be careful with the indices here!
                actions_on_next_level = num_actions_per_level[level + 1]
                argmax_idcs_considered = list(np.argpartition(considered_gumbels.numpy(), -actions_on_next_level)[-actions_on_next_level:])
                argmax_idcs_considered.sort()
                considered_actions = [considered_actions[idx] for idx in argmax_idcs_considered]

        # If we are done we choose from the remaining gumbels the final argmax action
        action = considered_actions[torch.argmax(considered_gumbels).item()]

        self.root.sequential_halving_chosen_action = action

        extra_info = {
            "max_search_depth": self.search_depth
        }

        return self.root, extra_info

    def run_simulations_for_considered_root_actions(self, considered_actions: List[int], num_simulations_per_action: int):
        """
        Performs "one level" of sequential halving, i.e. given a list of considered actions in the root,
        starts simulations for each of the considered actions multiple times.

        Parameters
        ----------
        considered_actions: [List[int]] Actions to visit in root.
        num_simulations_per_action: [int] How often to visit each of the considered actions.
        """
        for _ in range(num_simulations_per_action):
            inference_queries = dict()  # keeps track of queries on which to wait
            for action in considered_actions:
                # perform one search simulation starting from this action
                query_id, state, search_path, reward = self.run_single_simulation_from_root(for_action=action)
                if query_id is None:
                    # We do not need to wait for some inference and can immediately
                    # backpropagate
                    self.backpropagate(search_path)
                else:
                    inference_queries[query_id] = (state, search_path, reward)
            if len(inference_queries.keys()) > 0:
                # We have queries to wait for and nodes to expand. Collect
                # the results, expand the nodes and backpropagate.
                self.dispatch_prediction_queue()
                results = self.wait_for_prediction_results(list(inference_queries.keys()))
                for query_id in results:
                    state, search_path, reward = inference_queries[query_id]
                    policy_logits, value = results[query_id]
                    # expand node and backpropagate
                    value_baseline = None if self.value_estimates is None else self.value_estimates[search_path[-1].timestep]
                    search_path[-1].expand(policy_logits, value, reward, state, value_baseline)
                    self.backpropagate(search_path)

    def run_single_simulation_from_root(self, for_action: int) -> Tuple[Optional[str], BaseGame,
                                                                        List[SingleplayerGumbelNode], float]:
        """
        Runs a single simulation from the root taking the given action `for_action`.

        Parameters
        ----------
        for_action: [int] Action to take in root node.

        Returns:
            query_id [str]: Query id of the last node of search path is to be expaneded and needs
                to wait for prediction. Is `None` if no waiting for prediction is required.
            state [BaseGame]: Game state of last node in search path.
            search_path [List[SingleplayerGumbelNode]]: The full search path of the simulation.
        """
        node: SingleplayerGumbelNode = self.root.children[for_action]
        search_path = [self.root, node]
        action = for_action

        while node.expanded and not node.terminal:
            action, node = self.select_child(node)
            search_path.append(node)

        query_id = None
        state: Optional[BaseGame] = None
        reward = 0
        if not node.terminal:
            # now the current `node` is unexpanded, in particular it has no game state.
            # We expand it by copying the game of the parent and simulating a move.
            parent = search_path[-2]
            state = parent.state.copy()
            # simulate the move
            episode_done, reward = state.make_move(action)
            # if the game is over after simulating this move, we don't need a prediction from
            # the network. Simply call expand with None values
            if episode_done:
                node.expand(None, None, reward, state)
            else:
                # Otherwise we add the current state to the prediction queue
                query_id = self.add_state_to_prediction_queue(state)

        if len(search_path) > self.search_depth:
            self.search_depth = len(search_path)

        return query_id, state, search_path, reward

    def shift(self, action):
        """
        Shift tree to node by given action, making the node resulting from action the new root.

        A dirichlet_sample is then stored at this node to be used during MCTS
        """
        self.root: SingleplayerGumbelNode = self.root.children[action]
        self.root.parent_action = -1
        self.root.parent_node = None

    def backpropagate(self, search_path: List[SingleplayerGumbelNode]):
        """
        Backpropagates predicted value of the search path's last node through the
        search path and increments visit count for each node.
        """
        value = search_path[-1].predicted_value

        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value if not node.terminal else node.reward
            value = node.reward + value

    def select_child(self, node: SingleplayerGumbelNode) -> Tuple[int, SingleplayerGumbelNode]:
        """
        In-tree (non-root) action selection strategy as according to GAZ.

        Parameters
        ----------
        node: [SingleplayerGumbelNode] Node in which to select an action.

        Returns
        -------
            [int] Action to take in `node`.
            [SingleplayerGumbelNode] Resulting node when taking the selected action.
        """
        if len(node.children.items()) == 0:
            raise Exception(f"Gumbel MCTS `select_child`: Current node has no children.")

        # Otherwise we select the action using the completed Q values as stated in paper.
        improved_policy = self.get_improved_policy(node)
        action = torch.argmax(improved_policy - node.get_visit_count_distribution_tensor()).item()

        return action, node.children[action]

    def get_improved_policy(self, node: SingleplayerGumbelNode):
        """
        Given a node, computes the improved policy over the node's actions using the
        completed Q-values.
        """
        completed_q_values: torch.Tensor = node.get_completed_q_values()
        if self.min_max_normalization:
            self.min_max_stats.update(completed_q_values)
            completed_q_values = self.min_max_stats.normalize(completed_q_values)
        sigma_q_values = self.sigma_q(node, completed_q_values)
        improved_policy = torch.softmax(node.children_prior_logits + sigma_q_values, dim=0)
        return improved_policy

    def sigma_q(self, node: SingleplayerGumbelNode, q_values: torch.Tensor) -> torch.Tensor:
        """
        Monotonically increasing sigma function.

        Parameters
        ----------
        node: [GumbelNode] Node for whose actions the sigma function is computed.
        q_values: [torch.Tensor] Q-values for actions

        Returns
        -------
        [torch.Tensor] Element-wise evaluation of sigma function on `q_values`
        """
        max_visit = max([child.visit_count for child in node.children.values()])
        return (self.config.gumbel_c_visit + max_visit) * self.config.gumbel_c_scale * q_values


class MinMaxStats:
    """
    Holds the min-max values of the Q-values within the tree.
    """
    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")
        self.eps = 1e-8

    def update(self, q_values: torch.Tensor):
        self.maximum = max(self.maximum, torch.max(q_values).item())
        self.minimum = min(self.minimum, torch.min(q_values).item())

    def normalize(self, q_values: torch.Tensor):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (q_values - self.minimum) / max(self.eps, self.maximum - self.minimum)
        return q_values