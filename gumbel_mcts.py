import threading

import torch
import math
import time
import ray
import numpy as np

from typing import Optional, List, Dict, Tuple, Union

from inferencer import ModelInferencer
from local_inferencer import LocalInferencer
from base_game import BaseGame
from base_config import BaseConfig


class GumbelNode:
    """
    Represents one node in the search tree, i.e. the representation of a
    certain state, stemming from a parent state coupled with an action.
    """
    def __init__(self, prior: float, prior_logit: float, parent_node, parent_action: int):
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
        self.to_play = 0  # player which is on move in this node
        self.prior = prior
        self.prior_logit = prior_logit
        self.value_sum = 0  # sum of backpropagated estimated values. Corresponds to "W(s, a)" in the Alpha Zero paper
        self.children = {}  # mapping of action -> child node
        self.state: Optional[BaseGame] = None  # holds the state of the node as an instance copy of the game

        self.predicted_value = 0  # The value predicted by the network (resp. true value if game is terminal in this node)

        self.expanded = False
        self.terminal = False  # whether in this node the game is finished

        # keeps a torch.Tensor of the predicted logits for child actions for easier access
        self.children_prior_logits: Optional[torch.Tensor] = None

        # keeps track of the node's parent and which action led to this one
        self.parent_node = parent_node
        self.parent_action = parent_action

        # If the node is root at some point, holds the final chosen action after sequential halving
        self.sequential_halving_chosen_action: int = -1

        self.baseline_policy_logits = None

    def value(self) -> float:
        """
        Returns the state-action value of this node (i.e. Q(s, a) where `s` is parent state
        and `a` is action leading to this node, depending on the visit counts (corresponds to Q(s,a) in the Alpha
        Zero paper) from perspective of player on move.
        """
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, policy_logits: Optional[torch.Tensor], predicted_value: Optional[float], state: BaseGame) -> bool:
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

        self.to_play = self.state.get_current_player()

        finished, winner = self.state.is_finished_and_winner()
        if finished:
            self.terminal = True
            # set the predicted value to true value of the outcome
            self.predicted_value = 1 if winner == self.state.get_current_player() else -1
            return True

        self.children_prior_logits = policy_logits
        policy_probs = torch.softmax(policy_logits, dim=0).numpy().astype('float64')
        # normalize, in most cases the sum is not exactly equal to 1 which is problematic when sampling
        policy_probs /= policy_probs.sum()

        for action, p in enumerate(policy_probs):
            self.children[action] = GumbelNode(prior=p, prior_logit=policy_logits[action],
                                               parent_node=self, parent_action=action)
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
            sign = 1 if child.to_play == self.to_play else -1
            values.append(child.value() * sign)
            is_unvisited.append(True if child.visit_count == 0 else False)
        return torch.tensor(values), torch.tensor(is_unvisited)

    def get_completed_q_values(self) -> torch.Tensor:
        completed_q, unvisited_children = self.get_estimated_q_tensor()
        value_approximation = self.get_mixed_value_approximation()
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
                sign = 1. if node.to_play == self.to_play else -1.
                sum_visited_pi_q += pi * sign * node.value()

        mixed_value = self.predicted_value
        if sum_visited_pi != 0:
            mixed_value += (sum_visits / sum_visited_pi) * sum_visited_pi_q
        mixed_value /= 1. + sum_visits

        return mixed_value


class GumbelMCTS:
    """
    Core Monte Carlo Tree Search using Planning with Gumbel.
    The tree persists over the full game. We run N simulations using sequential halving at the root,
    and traverse the tree according to the "non-root" selection formula in Planning with Gumbel,
    which gets then expanded.
    """
    def __init__(self, actor_id: int, config: BaseConfig, model_inference_worker: Union[ModelInferencer, LocalInferencer],
                 game_options, network_class):
        """
        Parameters
            actor_id [int] Unique Identifier which is used to mark inference queries as belonging to this tree.
            config [BaseConfig]
            model_inference_worker: Inference worker
            game_options [GameOptions]: Game options for how MCTS should be performed for both players.
            network_module
        """
        self.id = actor_id
        self.config = config
        self.inference_worker = model_inference_worker
        self.game_options = game_options
        self.network_class = network_class

        self.root: GumbelNode = GumbelNode(prior=0, prior_logit=0, parent_node=None, parent_action=-1)

        # incrementing counter for simulations queried in this tree; used to identify returned queries
        self.query_counter = 0
        # Stores tuples of (actor_id, query_id, tensor, model_name) which need to be
        # sent to the model inference process.
        # In order not to send the leaf node queries for each simulation individually, we batch them
        # in each sequential halving level so to have only one roundtrip.
        self.model_keys = list(set(self.game_options["model_for_players"].values()))

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

        self.waiting_time = 0

        # Keeps track of full number of actions for a move when running at root
        self.num_actions_for_move = 0

    def add_state_to_prediction_queue(self, game_state: BaseGame) -> str:
        """
        Adds state to queue which is prepared for inference worker, and returns a
        query id which can be used to poll the results.
        """
        canonical_board = game_state.get_current_state()
        current_player = game_state.get_current_player()
        num_actions = len(game_state.get_actions())

        self.query_counter += 1
        query_id = str(self.query_counter)

        model_key = self.game_options["model_for_players"][current_player]
        self.query_states[model_key].append(canonical_board)
        self.query_ids[model_key].append(query_id)
        self.queries_num_actions[query_id] = num_actions

        return query_id

    def dispatch_prediction_queue(self):
        """
        Sends the current inferencing queue to the inference worker, if
        the queue is not empty. Empties the query list afterwards.
        """
        if not self.config.inference_on_experience_workers:
            # send states to central inferencer
            ray.get(self.inference_worker.add_list_to_queue.remote(self.id, self.query_ids, self.query_states, list(self.query_states.keys())))
            for model_key in self.model_keys:
                self.query_ids[model_key] = []
                self.query_states[model_key] = []
        else:
            # perform the inference directly on the experience worker on CPU
            for model_key in self.model_keys:
                query_ids = self.query_ids[model_key]
                if not len(query_ids):
                    continue
                policy_logits_batch, value_batch = self.inference_worker.infer_batch(self.query_states[model_key], model_key)
                for i, query_id in enumerate(query_ids):
                    policy_logits = torch.from_numpy(policy_logits_batch[i].copy())
                    value = value_batch[i][0]
                    self.query_results[query_id] = (policy_logits, value)

                self.query_ids[model_key] = []
                self.query_states[model_key] = []

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
        if "random_player" in self.game_options \
                and self.game_options["random_player"] is not None\
                and self.game_options["random_player"] == state.current_player:
            policy_logits = torch.ones_like(policy_logits)
            value = 0.

        self.root.expand(policy_logits, value, state)

    def run_at_root(self, game: BaseGame, n_actions: int) -> Tuple[GumbelNode, Dict]:
        self.num_actions_for_move = n_actions

        # Step 1: If the root is not expanded, we expand it
        if not self.root.expanded:
            self.expand_root(game)

        # Step 2: Check if the current player should simulate moves for policy improvement. If not, we return right after
        # expanding the root
        if not self.game_options["use_tree_simulations_for_players"][self.root.to_play] or n_actions == 1:
            return self.root, {}

        # Step 3: Sample `n_actions` Gumbel variables for sampling without replacement.
        if self.game_options["mcts_simulate_moves_deterministically"][self.root.to_play]:
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

    def run_simulations_for_root_actions_or_search_paths(self, root_actions: Optional[List[int]], search_paths: Optional[List[List[GumbelNode]]] = None):
        inference_queries = dict()  # keeps track of queries on which to wait
        use_predicted_baseline_logits = dict()
        counter = 1
        if root_actions is not None:
            # this is the regular case
            for action in root_actions:
                # perform one search simulation starting from this action
                query_id, state, search_path = self.run_single_simulation_from_root(for_action=action)
                if query_id is None:
                    # We do not need to wait for some inference and can immediately
                    # backpropagate
                    self.backpropagate(search_path)
                elif query_id == "use_predicted_baseline_logits":
                    use_predicted_baseline_logits[f"predicted_{counter}"] = (state, search_path)
                    counter += 1
                else:
                    inference_queries[query_id] = (state, search_path)
        elif search_paths is not None:
            # this is the case for when we continue simulations when the expanded leaf is the baseline player and we
            # need to expand the next node.
            for search_path in search_paths:
                # perform one search simulation starting from this action
                query_id, state, search_path = self.run_single_simulation_from_root(for_action=-1, continue_in_search_path=search_path)
                if query_id is None:
                    # We do not need to wait for some inference and can immediately
                    # backpropagate
                    self.backpropagate(search_path)
                else:
                    inference_queries[query_id] = (state, search_path)
        else:
            raise Exception("Either root_actions or search_paths must be given.")

        results = dict()
        if len(inference_queries.keys()) > 0:
            # We have queries to wait for and nodes to expand. Collect
            # the results, expand the nodes and backpropagate.
            self.dispatch_prediction_queue()
            results = self.wait_for_prediction_results(list(inference_queries.keys()))
        for key in use_predicted_baseline_logits:
            state, search_path = use_predicted_baseline_logits[key]
            results[key] = (search_path[-2].baseline_policy_logits, 0)  # value is set to zero

        rerun_for_search_paths = []
        for query_id in results:
            if "predicted" in query_id:
                state, search_path = use_predicted_baseline_logits[query_id]
            else:
                state, search_path = inference_queries[query_id]
            policy_logits, value = results[query_id]
            # expand node and backpropagate
            search_path[-1].expand(policy_logits, value, state)

            # if the expanded leaf is baseline player's turn, choose an action according to baseline policy,
            # and expand afterwards again
            leaf_player = search_path[-1].to_play
            if not self.game_options["use_tree_simulations_for_players"][leaf_player] and \
                    self.game_options["baseline_player"] is not None and \
                    leaf_player == self.game_options["baseline_player"]:
                rerun_for_search_paths.append(search_path)
                # also as we do not need to backpropagate the value of the baseline player, we can store
                # the policy_logits in the parent node, so it can be reused for other baseline actions which then
                # do not need to query the network.
                if search_path[-2].baseline_policy_logits is None:
                    search_path[-2].baseline_policy_logits = policy_logits
            else:
                self.backpropagate(search_path)

        if len(rerun_for_search_paths):
            self.run_simulations_for_root_actions_or_search_paths(root_actions=None, search_paths=rerun_for_search_paths)

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
            self.run_simulations_for_root_actions_or_search_paths(root_actions=considered_actions)

    def run_single_simulation_from_root(self, for_action: int, continue_in_search_path: Optional[List[GumbelNode]] = None) \
            -> Tuple[Optional[str], BaseGame, List[GumbelNode]]:
        """
        Runs a single simulation from the root taking the given action `for_action`.

        Parameters
        ----------
        for_action: [int] Action to take in root node.
        continue_in_search_path: Optional[List[GumbelNode]] If this is given, then the search continues in this
            search path and not from the root.

        Returns:
            query_id [str]: Query id of the last node of search path is to be expaneded and needs
                to wait for prediction. Is `None` if no waiting for prediction is required.
            state [JSPGame]: Game state of last node in search path.
            search_path [List[GumbelNode]]: The full search path of the simulation.
        """
        if continue_in_search_path is None:
            node: GumbelNode = self.root.children[for_action]
            search_path = [self.root, node]
            action = for_action
        else:
            search_path = continue_in_search_path
            node: GumbelNode = continue_in_search_path[-1]

        while node.expanded and not node.terminal:
            action, node = self.select_child(node)
            search_path.append(node)

        query_id = None
        state: Optional[BaseGame] = None
        if not node.terminal:
            # now the current `node` is unexpanded, in particular it has no game state.
            # We expand it by copying the game of the parent and simulating a move.
            parent = search_path[-2]
            state = parent.state.copy()
            # simulate the move
            state.make_move(action)
            # if the game is over after simulating this move, we don't need a prediction from
            # the network. Simply call expand with None values
            finished, _ = state.is_finished_and_winner()
            if finished:
                node.expand(None, None, state)
            elif not self.game_options["use_tree_simulations_for_players"][state.get_current_player()] and \
                    state.get_current_player() == self.game_options["baseline_player"] and \
                    parent.baseline_policy_logits is not None:
                # the node can be expanded from already predicted policy logits, as we do not need the value
                query_id = "use_predicted_baseline_logits"
            else:
                # Otherwise we add the current state to the prediction queue
                query_id = self.add_state_to_prediction_queue(state)

        if len(search_path) > self.search_depth:
            self.search_depth = len(search_path)

        return query_id, state, search_path

    def shift(self, action):
        """
        Shift tree to node by given action, making the node resulting from action the new root.

        A dirichlet_sample is then stored at this node to be used during MCTS
        """
        self.root: GumbelNode = self.root.children[action]
        self.root.parent_action = -1
        self.root.parent_node = None

    def backpropagate(self, search_path: List[GumbelNode]):
        """
        Backpropagates predicted value of the search path's last node through the
        search path and increments visit count for each node.
        """
        value = search_path[-1].predicted_value
        to_play = search_path[-1].to_play
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value if node.to_play == to_play else -value

    def select_child(self, node: GumbelNode) -> Tuple[int, GumbelNode]:
        """
        In-tree (non-root) action selection strategy as accoding to the Gumbel-Paper.

        Parameters
        ----------
        node: [GumbelNode] Node in which to select an action.

        Returns
        -------
            [int] Action to take in `node`.
            [GumbelNode] Resulting node when taking the selected action.
        """
        if len(node.children.items()) == 0:
            raise Exception(f"Gumbel MCTS `select_child`: Current node has no children.")

        to_play = node.to_play

        # Check if we should use the sophisticated action selection strategy, or rather
        # select action using only the priors.
        if not self.game_options["use_tree_simulations_for_players"][to_play]:
            actions = [action for action in node.children.keys()]
            priors = [child.prior for child in node.children.values()]
            if self.game_options["mcts_simulate_moves_deterministically"][to_play]:
                # Get action with maximum prior
                action = actions[priors.index(max(priors))]
            else:
                # Sample according to priors
                action = np.random.choice(actions, p=priors)

            return action, node.children[action]

        # Otherwise we select the action using the completed Q values as stated in paper.
        improved_policy = self.get_improved_policy(node)
        action = torch.argmax(improved_policy - node.get_visit_count_distribution_tensor()).item()

        return action, node.children[action]

    def get_improved_policy(self, node: GumbelNode):
        """
        Given a node, computes the improved policy over the node's actions using the
        completed Q-values.
        """
        completed_q_values: torch.Tensor = node.get_completed_q_values()
        sigma_q_values = self.sigma_q(node, completed_q_values)
        improved_policy = torch.softmax(node.children_prior_logits + sigma_q_values, dim=0)
        return improved_policy

    def sigma_q(self, node: GumbelNode, q_values: torch.Tensor) -> torch.Tensor:
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

    @staticmethod
    def get_sequential_halving_simulations_for_levels(num_actions: int, simulation_budget: int) -> Tuple[
        List[int], List[int]]:
        """
        Given a number of actions and a simulation budget calculates how many simulations
        in each sequential-halving-level may be used for each action.

        Returns:
            List[int] Number of actions for each level.
            List[int] On each level, number of simulations which can be spent on each action.
        """
        num_simulations_per_action = []
        actions_on_levels = []

        # number of levels if simulations
        num_levels = math.floor(math.log2(num_actions))

        remaining_actions = num_actions
        remaining_budget = simulation_budget
        for level in range(num_levels):
            if level > 0:
                remaining_actions = max(2, math.floor(remaining_actions / 2))

            if remaining_budget < remaining_actions:
                break

            actions_on_levels.append(remaining_actions)
            num_simulations_per_action.append(
                max(1, math.floor(simulation_budget / (num_levels * remaining_actions)))
            )
            remaining_budget -= num_simulations_per_action[-1] * actions_on_levels[-1]

        if remaining_budget > 0:
            num_simulations_per_action[-1] += remaining_budget // actions_on_levels[-1]

        return actions_on_levels, num_simulations_per_action

