import torch
from torch import nn
from base_config import BaseConfig
from typing import List, Dict
from model.transformer_modules import TransformerBlock, AttentionNarrow
from model.base_network import BaseNetwork, FeedForward
import gaz_ptp.jssp_network as jssp_network


class Network(BaseNetwork):
    def __init__(self, config: BaseConfig, device: torch.device = None):
        super().__init__(config, device)

        self.config = config
        self.device = torch.device("cpu") if device is None else device

        self.latent_dimension_operation_seq = self.config.problem_specifics["latent_dimension_operation_seq"]
        self.latent_dimension_job_seq = self.config.problem_specifics["latent_dimension_job_seq"]

        self.situation_net = jssp_network.SituationNetwork(config=self.config, device=self.device)
        self.prediction_net = PredictionNetwork(config=self.config, device=self.device)

    def forward(self, x: Dict):
        """
        Parameters
        ----------
        x: [Dict] Dictionary with input values as returned from `states_to_batch`.

        Returns
        -------
            policy_logits_padded: [torch.Tensor] Policy logits which are padded to the maximum number of remaining nodes
                in the batch. Shape (batch_size, <maximum remaining nodes sequence length>)
            values: [torch.Tensor] Values as tensor of shape (batch_size, 1)
        """
        boards = x
        situation_vector_batch, jobs_transformed_batch = self.situation_net(boards["current"])

        policy_logits_padded_masked, values = self.prediction_net(
            situation_vector=situation_vector_batch,
            jobs_transformed=jobs_transformed_batch,
            baseline=boards["current"]["baseline_batch"],
            policy_attention_mask=boards["current"]["policy_mask_batch_tensor"]
        )

        return policy_logits_padded_masked, values

    @staticmethod
    def states_to_batch(states: List, config: BaseConfig, to_device: torch.device = None):
        """
        Given a list of canonical boards, each of which is itself a 1-element list with the player's situation as
        returned from `get_current_state`, prepares all the tensors and index lists needed for a batched forward pass.
        We denote by `current` the player to be compatible with GAZ PTP.
        """
        boards = states
        return {
            "current": Network.player_situations_to_batch(
                situations=[board[0] for board in boards], config=config, to_device=to_device,
                custom_max_job_seq_len=config.problem_specifics["num_jobs"]
            )
        }

    @staticmethod
    def player_situations_to_batch(situations: List[Dict], config: BaseConfig, to_device: torch.device = None,
                                   custom_max_job_seq_len: int = 0):
        batch_dict = jssp_network.Network.player_situations_to_batch(situations, config, to_device, custom_max_job_seq_len)

        # only add the scalar baseline to batch
        baseline_batch = torch.tensor([situation["baseline_outcome"] for situation in situations]) \
            .float().unsqueeze(-1).to(to_device)

        batch_dict["baseline_batch"] = baseline_batch
        return batch_dict

    @staticmethod
    def states_batch_dict_to_device(batch_dict: Dict, to_device: torch.device):
        return jssp_network.Network.states_batch_dict_to_device(batch_dict, to_device)


class PredictionNetwork(nn.Module):

    def __init__(self, config: BaseConfig, device: torch.device = None):
        super().__init__()

        self.config = config
        self.device = torch.device("cpu") if device is None else device
        self.has_baseline = self.config.singleplayer_options is not None \
                            and self.config.singleplayer_options["method"] == "greedy_scalar"

        self.latent_dimension = self.config.problem_specifics["latent_dimension_job_seq"]

        # Attention layer which is applied to the feedforwarded, attended board vector to get the policy.
        self.policy_attention = AttentionNarrow(query_dim=self.latent_dimension,
                                                key_dim=self.latent_dimension,
                                                latent_dim=self.latent_dimension,
                                                n_attention_heads=1,
                                                use_query_linear=True
                                                )

        # Feedforward to get value from board vector (concatenated situations of both players)
        input_dim = self.latent_dimension + 1 if self.has_baseline else self.latent_dimension
        self.value_feedforward = nn.Sequential(
            nn.Linear(input_dim, 2 * self.latent_dimension),
            nn.GELU(),
            nn.Linear(2 * self.latent_dimension, 2 * self.latent_dimension),
            nn.GELU(),
            nn.Linear(2 * self.latent_dimension, 2 * self.latent_dimension),
            nn.GELU(),
            nn.Linear(2 * self.latent_dimension, 1)
        )

    def forward(self, situation_vector, jobs_transformed, baseline,
                policy_attention_mask):
        batch_size = situation_vector.shape[0]

        if self.has_baseline:
            # Use the board vector to obtain the values in [-1, 1] via a feedforward
            values = self.value_feedforward(torch.cat([situation_vector, baseline], dim=1))
            values = torch.tanh(values)
        else:
            # regular singleplayer linear output
            values = self.value_feedforward(situation_vector)

        # Use the transformed board vector as query to perform a one-headed attention pass and use the attention weights
        # as basis for the policy computation
        # The attention weights are padded to the maximum remaining nodes sequence
        _, attention_weights_unmasked = self.policy_attention(
            query=situation_vector,
            keys=jobs_transformed,
            padding_mask=policy_attention_mask
        )

        # As in "Attention, Learn to solve routing problems" we apply a tanh function and scale to [-C, C]
        # with C=10, which is then used as logits
        attention_weights_unmasked = attention_weights_unmasked.view(batch_size, -1)
        board_attention_mask = policy_attention_mask.view(batch_size, -1)
        policy_logits_padded = 10. * torch.tanh(attention_weights_unmasked)
        # Now mask the policy_logits_padded, that the padding won't affect the softmax later
        policy_logits_padded_masked = policy_logits_padded + (1. - board_attention_mask) * -10000.

        return policy_logits_padded_masked, values
