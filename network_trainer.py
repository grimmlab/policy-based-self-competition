import math

import torch
import os
import numpy as np
import ray
import time
import copy

from model.base_network import BaseNetwork, dict_to_cpu
from base_config import BaseConfig

from typing import Dict, Type

from shared_storage import SharedStorage


@ray.remote
class NetworkTrainer:
    """
    One instance of this class runs in a separate process, continuosly training the SynGameMuZeroNetwork using the
    experience sampled from the playing actors and saving the weights in the shared storage.
    """
    def __init__(self, config: BaseConfig, shared_storage: SharedStorage, network_class: Type[BaseNetwork],
                 initial_checkpoint: Dict = None, device: torch.device = None):
        self.config = config
        self.device = device if device else torch.device("cpu")
        self.shared_storage = shared_storage
        self.uses_arena = True if (self.config.singleplayer_options is None
                                   or self.config.singleplayer_options["method"] in ["greedy_scalar", "single_timestep"]) else False

        if self.config.CUDA_VISIBLE_DEVICES:
            # override ray's limiting of GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config.CUDA_VISIBLE_DEVICES

        # Fix random generator seed
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initialize the network
        self.model = network_class(config, device=self.device)
        if initial_checkpoint["weights_newcomer"] is not None:
            self.model.set_weights(copy.deepcopy(initial_checkpoint["weights_newcomer"]))
        else:
            # If we do not have an initial checkpoint, we set the random weights both to 'newcomer' and 'best'.
            print("Setting identical random weights to 'newcomer' and 'best' model...")
            self.shared_storage.set_info.remote({
                "weights_timestamp_newcomer": round(time.time() * 1000),
                "weights_timestamp_best": round(time.time() * 1000),
                "weights_newcomer": copy.deepcopy(self.model.get_weights()),
                "weights_best": copy.deepcopy(self.model.get_weights())
            })

        self.model.to(self.device)
        self.model.train()

        self.training_step = initial_checkpoint["training_step"] if initial_checkpoint else 0

        if "cuda" not in str(next(self.model.parameters()).device):
            print("NOTE: You are not training on GPU.\n")

        # Initialize the optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr_init,
            weight_decay=self.config.weight_decay
        )

        # Load optimizer state if available
        if initial_checkpoint is not None and initial_checkpoint["optimizer_state"] is not None:
            print("Loading optimizer...\n")
            self.optimizer.load_state_dict(
                copy.deepcopy(initial_checkpoint["optimizer_state"])
            )

    def continuous_update_weights(self, replay_buffer, logger=None):
        """
        Continuously samples batches from the replay buffer, make an optimization step, repeat...
        """

        # Wait for replay buffer to contain at least a certain number of games.
        while ray.get(replay_buffer.get_length.remote()) < max(1, self.config.start_train_after_episodes):
            time.sleep(1)

        next_batch_policy = replay_buffer.get_batch.remote(for_value=False)
        next_batch_value = replay_buffer.get_batch.remote(for_value=True)

        # Main training loop
        while not ray.get(self.shared_storage.get_info.remote("terminate")):
            # If we should pause, sleep for a while and then continue
            if ray.get(self.shared_storage.in_evaluation_mode.remote()):
                time.sleep(1)
                continue

            batch_policy = ray.get(next_batch_policy)
            batch_value = ray.get(next_batch_value)

            # already prepare next batch in the replay buffer worker, so we minimize waiting times
            next_batch_policy = replay_buffer.get_batch.remote(for_value=False)
            next_batch_value = replay_buffer.get_batch.remote(for_value=True)

            # perform exponential learning rate decay based on training steps. See config for more info
            self.update_lr()
            self.training_step += 1

            # loss for this batch
            policy_loss = self.get_loss(batch_policy, for_value=False)
            value_loss = self.get_loss(batch_value, for_value=True)

            # combine policy and value loss
            loss = policy_loss + self.config.value_loss_weight * value_loss
            loss = loss.mean()

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()

            if self.config.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.gradient_clipping)

            self.optimizer.step()

            # Save model to shared storage so it can be used by the actors
            if self.training_step % self.config.checkpoint_interval == 0:
                self.shared_storage.set_info.remote({
                    "weights_timestamp_newcomer": round(time.time() * 1000),
                    "weights_newcomer": copy.deepcopy(self.model.get_weights()),
                    "optimizer_state": copy.deepcopy(
                        dict_to_cpu(self.optimizer.state_dict())
                    )
                })

            # Send results to logger
            if logger is not None:
                logger.training_step.remote({
                    "loss": loss.item(), "value_loss": value_loss.mean().item(), "policy_loss": policy_loss.mean().item()
                })

            # Check for arena mode
            if self.uses_arena and (self.training_step % self.config.arena_checkpoint_interval == 0):
                ray.get(self.shared_storage.set_evaluation_mode.remote(True))
                # Let's test if the new model is better than the currently best performing
                arena_results = self.model_arena()

                newcomer_wins_arena = False
                if self.config.arena_criteria_win_ratio > 0:
                    if (1. * arena_results["newcomer_num_wins"]) / (arena_results["newcomer_num_wins"] + arena_results["best_num_wins"]) > self.config.arena_criteria_win_ratio:
                        newcomer_wins_arena = True
                elif arena_results["avg_objective_margin"] > 1e-5:
                    newcomer_wins_arena = True

                if newcomer_wins_arena:
                    # Set the current newcomer model as the new best model.
                    # Also if "best" has been playing randomly, this is set to False.
                    print("Replacing current best with newcomer model...")
                    self.shared_storage.set_info.remote({
                        "weights_timestamp_best": round(time.time() * 1000),
                        "weights_best": copy.deepcopy(self.model.get_weights()),
                        "best_plays_randomly": False
                    })

                logger.arena_step.remote(arena_results)

                ray.get(self.shared_storage.set_evaluation_mode.remote(False))

            # Inform shared storage of training step. We do this at the end so there are no conflicts with
            # the arena mode.
            self.shared_storage.set_info.remote({
                "training_step": self.training_step
            })

            # Managing the episode / training ratio

            if self.config.ratio_range:
                infos: Dict = ray.get(
                    self.shared_storage.get_info.remote(["training_step", "num_played_games", "terminate"]))
                ratio = infos["training_step"] / max(1, infos["num_played_games"] - self.config.start_train_after_episodes)

                while (ratio > self.config.ratio_range[1]
                       and not infos["terminate"] and not ray.get(self.shared_storage.in_evaluation_mode.remote())
                ):
                    infos: Dict = ray.get(
                        self.shared_storage.get_info.remote(["training_step", "num_played_games", "terminate"])
                    )
                    ratio = infos["training_step"] / max(1, infos["num_played_games"] - self.config.start_train_after_episodes)
                    #print(infos["training_step"], max(1, infos["num_played_games"]), ratio)
                    time.sleep(0.010)  # wait for 10ms

    def model_arena(self) -> Dict:
        print("Arena against best performing model...")
        # Load problem instances
        problem_instances = np.load(self.config.arena_set_path)[:self.config.num_arena_games]
        problem_instances = [(i, problem_instances[i], "arena") for i in range(self.config.num_arena_games)]

        ray.get(self.shared_storage.set_to_evaluate.remote(problem_instances))

        eval_results = [None] * self.config.num_arena_games

        num_wins_newcomer = 0
        num_wins_best = 0
        sum_objective_newcomer = 0
        sum_objective_best = 0
        objective_margin = 0  # sum of objective difference

        # collect results
        while None in eval_results:
            time.sleep(0.1)
            fetched_results = ray.get(self.shared_storage.fetch_evaluation_results.remote())
            for (i, result) in fetched_results:
                eval_results[i] = result
                objective_best = result["objectives"][-1]
                sum_objective_best += objective_best
                sum_objective_newcomer += result["objectives"][1]
                # check if it is a draw. we do not count draws
                abs_difference_obj = math.fabs(result["objectives"][1] - result["objectives"][-1])
                if abs_difference_obj < 1e-5:
                    continue
                if result["objectives"][1] < result["objectives"][-1]:
                    num_wins_newcomer += 1
                    objective_margin += abs_difference_obj
                else:
                    num_wins_best += 1
                    objective_margin -= abs_difference_obj

        avg_objective_best = sum_objective_best / self.config.num_arena_games
        avg_objective_newcomer = sum_objective_newcomer / self.config.num_arena_games
        objective_margin /= self.config.num_arena_games

        result_dict = {
            "newcomer_num_wins": num_wins_newcomer,
            "best_num_wins": num_wins_best,
            "avg_objective_newcomer": avg_objective_newcomer,
            "avg_objective_best": avg_objective_best,
            "avg_objective_margin": objective_margin
        }

        return result_dict

    def get_loss(self, batch, for_value=False):
        """
        Parameters:
            for_value [bool]: If True, value loss is returned, else policy loss.

        Returns:
            [torch.Tensor] of shape (batch size,)
        """
        (
            state_batch,  # observation
            value_batch_tensor,  # (batch_size, 1)
            policy_batch_tensor,  # (batch_size, <max policy length in batch>)
            policy_averaging_tensor
        ) = batch

        # Send everything to device
        state_batch = self.model.states_batch_dict_to_device(state_batch, self.device)

        # extract the query mask from the board batch for proper masking of padded policies
        value_batch_tensor = value_batch_tensor.to(self.device)
        policy_batch_tensor = policy_batch_tensor.to(self.device)
        policy_averaging_tensor = policy_averaging_tensor.to(self.device)

        # Generate predictions
        (
            policy_logits_padded,
            predicted_value_batch_tensor
        ) = self.model(state_batch)

        # Compute loss for each step
        value_loss, policy_loss = self.loss_function(
            predicted_value_batch_tensor, policy_logits_padded,
            value_batch_tensor, policy_batch_tensor, policy_averaging_tensor,
            use_kl=not self.config.gumbel_simple_loss
        )

        # Scale value loss
        if for_value:
            return value_loss
        else:
            return policy_loss

    @staticmethod
    def loss_function(value, policy_logits_padded, target_value, target_policy_tensor, policy_averaging_tensor, use_kl=False):
        """
        Parameters
            value: Tensor of shape (batch_size, 1)
            policy_logits_padded: Policy logits which are padded to have the same size.
                Tensor of shape (batch_size, <maximum policy size>)
            target_value: Tensor of shape (batch_size, 1)
            target_policy_tensor: Tensor of shape (batch_size, <max policy len in batch>)

        Returns
            value_loss, policy_loss
        """
        value_loss = torch.square(value - target_value).sum(dim=1)

        # Apply log softmax to the policy, and mask the padded values to 0.
        log_softmax = torch.nn.LogSoftmax(dim=1)
        log_softmax_policy_masked = log_softmax(policy_logits_padded)

        if not use_kl:
            # Cross entropy loss between target distribution and predicted one
            policy_loss = torch.sum(- target_policy_tensor * log_softmax_policy_masked, dim=1)
        else:
            # Kullback-Leibler
            kl_loss = torch.nn.KLDivLoss(reduction='none')
            policy_loss = kl_loss(log_softmax_policy_masked, target_policy_tensor)
            policy_loss = torch.sum(policy_loss, dim=1)

        # Average policy loss element-wise by length of individual policies
        #policy_loss = torch.div(policy_loss.unsqueeze(-1), policy_averaging_tensor)

        return value_loss, policy_loss

    def update_lr(self):
        """
        Update learning rate with an exponential scheme.
        """
        lr = self.config.lr_init * self.config.lr_decay_rate ** (self.training_step / self.config.lr_decay_steps)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
