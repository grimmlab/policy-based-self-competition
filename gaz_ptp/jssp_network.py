import torch
from torch import nn
from base_config import BaseConfig
from typing import List, Dict
from model.transformer_modules import TransformerBlock, AttentionNarrow
from model.base_network import BaseNetwork, FeedForward


class Network(BaseNetwork):
    def __init__(self, config: BaseConfig, device: torch.device = None):
        super().__init__(config, device)

        self.config = config
        self.device = torch.device("cpu") if device is None else device

        self.latent_dimension_operation_seq = self.config.problem_specifics["latent_dimension_operation_seq"]
        self.latent_dimension_job_seq = self.config.problem_specifics["latent_dimension_job_seq"]

        self.situation_net = SituationNetwork(config=self.config, device=self.device)
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
        situation_vectors = []
        jobs_transformed_list = []

        # We fold the opponent board into the batch dimension so that we can compute the situation vectors
        # for both players in one pass.
        batch_size = boards["current"]["operation_machine_idcs_tensor"].shape[0]
        combined_board = dict()
        for key in boards["current"]:
            if torch.is_tensor(boards["current"][key]):
                combined_board[key] = torch.cat(
                    (boards["current"][key], boards["opponent"][key]),
                    dim=0
                )

        situation_vector_batch, jobs_transformed_batch = self.situation_net(combined_board)
        situation_vectors.append(situation_vector_batch[:batch_size])  # for player 1
        situation_vectors.append(situation_vector_batch[batch_size:])  # for player -1
        jobs_transformed_list.append(jobs_transformed_batch[:batch_size])
        jobs_transformed_list.append(jobs_transformed_batch[batch_size:])

        policy_logits_padded_masked, values = self.prediction_net(
            situation_vector_current=situation_vectors[0],
            situation_vector_opponent=situation_vectors[1],
            jobs_transformed=jobs_transformed_list[0],
            policy_attention_mask=boards["current"]["policy_mask_batch_tensor"]
        )

        return policy_logits_padded_masked, values

    @staticmethod
    def states_to_batch(states: List, config: BaseConfig, to_device: torch.device = None):
        """
        Given a list of canonical boards, each of which is itself a 2-element list with the player's situation as
        returned from `get_canonical_board`, prepares all the tensors and index lists needed for a batched forward pass.
        For historical reasons we denote by `current` player 1, and by `opponent` player 2.
        """
        # Get the maximum job sequence length of all remaining job sequences, as we are later batching current and
        # opponent situations for faster forward pass.
        # max_seq_len = max([max(len(board[0]["num_ops_per_job"]), len(board[1]["num_ops_per_job"]))
        #                   for board in boards])
        boards = states
        return {
            "current": Network.player_situations_to_batch(
                situations=[board[0] for board in boards], config=config, to_device=to_device,
                custom_max_job_seq_len=config.problem_specifics["num_jobs"]
            ),
            "opponent": Network.player_situations_to_batch(
                situations=[board[1] for board in boards], config=config, to_device=to_device,
                custom_max_job_seq_len=config.problem_specifics["num_jobs"]
            )
        }

    @staticmethod
    def player_situations_to_batch(situations: List[Dict], config: BaseConfig, to_device: torch.device = None,
                                   custom_max_job_seq_len: int = 0):
        if not to_device:
            to_device = torch.device("cpu")

        batch_size = len(situations)

        max_num_jobs = max([situation["num_remaining_jobs"] for situation in situations]) \
            if not custom_max_job_seq_len else custom_max_job_seq_len

        # --- Prepare everything for the sequence of operations, which will result in a job embedding

        # pad operation machine idcs and stack them to shape (batch, max jobs, num ops)
        operation_machine_idcs = torch.stack([
            nn.functional.pad(situation["operation_machine_idcs"],
                              pad=(0, 0, 0, max_num_jobs - situation["num_remaining_jobs"]))
            for situation in situations
        ]).to(to_device)

        # pad operation processing times and stack them to shape (batch, max jobs, num ops, 1)
        operation_processing_times = torch.stack([
            nn.functional.pad(situation["operation_processing_times"],
                              pad=(0, 0, 0, max_num_jobs - situation["num_remaining_jobs"]))
            for situation in situations
        ]).unsqueeze(-1).to(to_device)

        machine_availability = torch.stack([
            situation["machine_availability_tensor"] for situation in situations
        ]).to(to_device)  # (batch, num machines)

        job_availability = torch.stack([
            nn.functional.pad(situation["job_availability_tensor"],
                              pad=(0, max_num_jobs - situation["num_remaining_jobs"]))
            for situation in situations
        ]).unsqueeze(-1).to(to_device)  # (batch, max num jobs, 1)

        # Get padding for operation sequences (+3 for situation token, job availability, machine availabilities)
        additional_tokens = 3
        ops_sequence_mask = torch.ones((batch_size, max_num_jobs, config.problem_specifics["num_operations"] + additional_tokens), dtype=torch.float32)
        for i, situation in enumerate(situations):
            for j in range(situation["num_remaining_jobs"]):
                num_ops = situation["num_ops_per_job"][j]
                ops_sequence_mask[i, j, num_ops + additional_tokens:] = 0.
        ops_sequence_mask = ops_sequence_mask.to(to_device)

        # -- Prepare everything for the job sequences
        num_jobs_tensor = torch.FloatTensor([
            situation["num_remaining_jobs"]
            for situation in situations
        ]).unsqueeze(-1).to(to_device)  # shape (batch, 1)

        min_absolute_machine_time = torch.FloatTensor([
            situation["min_machine_time"]
            for situation in situations
        ]).unsqueeze(-1).to(to_device)  # shape (batch, 1)

        # padding for the job tensor (+4 for situation token, min machine time, num jobs, machine
        # availabilities)
        additional_tokens = 4
        job_sequence_mask = torch.ones(batch_size, max_num_jobs + additional_tokens, dtype=torch.float32)
        # mask for the policy, i.e. only the transformed jobs. Here the max sequence length is without the 4
        # extra tokens
        policy_masks_batch = torch.ones((batch_size, max_num_jobs), dtype=torch.float32)
        for i, situation in enumerate(situations):
            job_sequence_mask[i, situation["num_remaining_jobs"] + additional_tokens:] = 0
            policy_masks_batch[i, situation["num_remaining_jobs"]:] = 0

        job_sequence_mask = job_sequence_mask.to(to_device)
        policy_masks_batch = policy_masks_batch.to(to_device)
        policy_masks_batch = policy_masks_batch.view(batch_size, 1, 1, max_num_jobs)
        # mask which is applied to attentions, i.e. masking tensors of size (batch, heads, queries, seq_len)

        return {
            "operation_machine_idcs_tensor": operation_machine_idcs,  # (batch, max jobs, num ops)
            "operation_processing_times_tensor": operation_processing_times,  # (batch, max jobs, num ops, 1)
            "machine_availability_tensor": machine_availability,  # (b, num machines)
            "job_availability_tensor": job_availability,  # (b, num_jobs, 1)
            "ops_sequence_mask_tensor": ops_sequence_mask,  # (b, <max num jobs>, num ops + 3)
            "num_jobs_tensor": num_jobs_tensor,  # (b, 1)
            "min_absolute_machine_time_tensor": min_absolute_machine_time,  # (b, 1)
            "job_sequence_mask_tensor": job_sequence_mask,  # (b, max num jobs + 4)
            "policy_mask_batch_tensor": policy_masks_batch,  # (b, 1, 1, max num jobs)
            "policy_padded_to": max_num_jobs
        }

    @staticmethod
    def states_batch_dict_to_device(batch_dict: Dict, to_device: torch.device):
        for i in batch_dict:
            for key in batch_dict[i]:
                if torch.is_tensor(batch_dict[i][key]):
                    batch_dict[i][key] = batch_dict[i][key].to(to_device)
        return batch_dict


class SituationNetwork(torch.nn.Module):
    """
    Network which computes a latent representation of the board situation of one player.
    """
    def __init__(self, config: BaseConfig, device: torch.device = None):
        super().__init__()

        self.config = config
        self.device = torch.device("cpu") if device is None else device

        self.latent_dimension_operation_seq = self.config.problem_specifics["latent_dimension_operation_seq"]
        self.latent_dimension_pre_operation_seq = self.config.problem_specifics["latent_dimension_pre_operation_seq"]
        self.latent_dimension_job_seq = self.config.problem_specifics["latent_dimension_job_seq"]
        self.num_jobs = self.config.problem_specifics["num_jobs"]
        self.num_machines = self.config.problem_specifics["num_machines"]
        self.num_operations = self.config.problem_specifics["num_operations"]

        # For the sequence of operations resulting in embeddings for each job
        self.machine_idx_lookup = nn.Embedding(self.num_machines, embedding_dim=self.latent_dimension_pre_operation_seq)
        self.operation_position_lookup = nn.Embedding(self.num_operations, embedding_dim=self.latent_dimension_pre_operation_seq)

        # Linear projection of machine idx, operation position and processing time
        self.operation_combined_linear = nn.Linear(self.latent_dimension_pre_operation_seq * 2 + 1, self.latent_dimension_operation_seq)

        self.job_token_lookup = nn.Embedding(1, embedding_dim=self.latent_dimension_operation_seq)
        self.machine_availability_op_seq_linear = nn.Linear(self.num_machines, self.latent_dimension_operation_seq)
        self.job_availability_linear = nn.Linear(1, self.latent_dimension_operation_seq)

        self.operation_transformer_blocks = nn.ModuleList([])

        for _ in range(self.config.problem_specifics["num_transformer_blocks_operation_seq"]):
            self.operation_transformer_blocks.append(
                TransformerBlock(
                    latent_dim=self.latent_dimension_operation_seq,
                    n_attention_heads=self.config.problem_specifics["num_attention_heads_operation_trf"],
                    feedforward_hidden_dim_mult=4,
                    dropout=0.0,
                    normalization="sequence",
                    use_mask=True, use_attention_bias=False
                )
            )

        self.job_embedding_linear = nn.Linear(self.latent_dimension_operation_seq, self.latent_dimension_job_seq, bias=True)


        # --- After having an embedding of each job, we compute a representation of the full
        # problem instance

        # Embedding of maketimes so far
        self.global_min_machine_time_linear = nn.Linear(1, self.latent_dimension_job_seq, bias=True)
        self.machine_availability_linear = nn.Linear(self.num_machines, self.latent_dimension_job_seq, bias=True)
        self.number_of_jobs_linear = nn.Linear(1, self.latent_dimension_job_seq, bias=True)

        # The situation token is an additional element in the sequence which gets piped through the transformer
        # The transformed token is used as a latent representation of the situation. This is comparable to a <cls>-token
        # in NLP
        self.situation_token_lookup = nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dimension_job_seq)

        # The transformer for embedding of job sequences (i.e. getting representation of the problem instance)
        job_seq_transformer_blocks = []
        for _ in range(self.config.problem_specifics["num_transformer_blocks_job_seq"]):
            job_seq_transformer_blocks.append(
                TransformerBlock(
                    latent_dim=self.latent_dimension_job_seq,
                    n_attention_heads=self.config.problem_specifics["num_attention_heads"],
                    feedforward_hidden_dim_mult=4,
                    dropout=0.0,
                    normalization="sequence",
                    use_mask=True, use_attention_bias=False
                )
            )
        self.job_seq_transformer = nn.ModuleList(job_seq_transformer_blocks)

    def forward(self, board_situation):
        """
        Parameters
        ----------
        board_situation: [Dict] Situation batch dictionary as returned
            from `JSPTransformerNetwork.player_situations_to_batch`

        Returns
        -------
            [torch.Tensor] Situation vector as tensor of shape (batch, latent_dimension_job_seq)
            [torch.Tensor] Transformed jobs to choose from as tensor
                of shape (batch, <max num of jobs>, latent_dimension_job_seq)
        """
        batch_size, num_jobs, num_ops = board_situation["operation_machine_idcs_tensor"].shape

        ops_machine_idx_embedding = self.machine_idx_lookup(board_situation["operation_machine_idcs_tensor"])\
            .view(batch_size * num_jobs, num_ops, self.latent_dimension_pre_operation_seq)

        ops_position_idcs = torch.LongTensor(list(range(num_ops))).to(self.device).unsqueeze(0)
        ops_position_idcs = ops_position_idcs.repeat(batch_size * num_jobs, 1).view(batch_size * num_jobs * num_ops)
        ops_position_embedding_ = self.operation_position_lookup(ops_position_idcs)
        ops_position_embedding = ops_position_embedding_.view(batch_size * num_jobs, num_ops,
                                                              self.latent_dimension_operation_seq)

        # concatenate machine and position embedding with processing time and project linearly
        operation_combined = torch.cat([
            ops_machine_idx_embedding,
            ops_position_embedding,
            board_situation["operation_processing_times_tensor"].view(batch_size * num_jobs, num_ops, 1)
        ], dim=-1)

        # prepare tokens for operation sequence transformer
        operation_embedding = self.operation_combined_linear(operation_combined)
        machine_availability_tensor = board_situation["machine_availability_tensor"]
        machine_availability_jobwise = machine_availability_tensor.unsqueeze(1).repeat(1, num_jobs, 1)
        machine_availability_jobwise = machine_availability_jobwise.view(batch_size * num_jobs, -1)
        machine_availability_for_ops_seq = self.machine_availability_op_seq_linear(machine_availability_jobwise)
        job_availability_embedding = self.job_availability_linear(board_situation["job_availability_tensor"]).view(batch_size * num_jobs, self.latent_dimension_operation_seq)
        job_token = self.job_token_lookup(
            torch.LongTensor([0] * (batch_size * num_jobs)).to(self.device)
        )

        operation_sequence = torch.cat([
            job_token.unsqueeze(1),
            machine_availability_for_ops_seq.unsqueeze(1),
            job_availability_embedding.unsqueeze(1),
            operation_embedding
        ], dim=1)

        # Pipe through transformer
        operation_sequence_transformed = operation_sequence
        operation_sequence_mask = board_situation["ops_sequence_mask_tensor"].view(batch_size * num_jobs, num_ops + 3)
        operation_sequence_mask = operation_sequence_mask.view(batch_size * num_jobs, 1, 1, num_ops + 3)

        for trf_block in self.operation_transformer_blocks:
            operation_sequence_transformed = trf_block(operation_sequence_transformed,
                                                       padding_mask=operation_sequence_mask)

        # project the transformed job tokens for the full schedule sequence
        job_embedding_sequence = self.job_embedding_linear(operation_sequence_transformed[:, 0, :])\
            .view(batch_size, num_jobs, self.latent_dimension_job_seq)
        num_jobs_embedding = self.number_of_jobs_linear(board_situation["num_jobs_tensor"])
        global_maketime_embedding = self.global_min_machine_time_linear(board_situation["min_absolute_machine_time_tensor"])
        machine_availability_embedding = self.machine_availability_linear(board_situation["machine_availability_tensor"])

        # Get the situation token which will result in our latent situation
        situation_token = self.situation_token_lookup(
            torch.LongTensor([0] * batch_size).to(self.device)
        )

        job_sequence = torch.cat([
            situation_token.unsqueeze(1),
            num_jobs_embedding.unsqueeze(1),
            global_maketime_embedding.unsqueeze(1),
            machine_availability_embedding.unsqueeze(1),
            job_embedding_sequence
        ], dim=1)

        # Pipe through transformer
        job_sequence_transformed = job_sequence
        job_sequence_mask = board_situation["job_sequence_mask_tensor"]
        job_padding_mask = job_sequence_mask.view(batch_size, 1, 1, num_jobs + 4)

        for trf_block in self.job_seq_transformer:
            job_sequence_transformed = trf_block(job_sequence_transformed,
                                                 padding_mask=job_padding_mask)

        situation_token = job_sequence_transformed[:, 0, :]
        jobs_transformed = job_sequence_transformed[:, 4:, :]

        return situation_token, jobs_transformed


class PredictionNetwork(nn.Module):

    def __init__(self, config: BaseConfig, device: torch.device = None):
        super().__init__()

        self.config = config
        self.device = torch.device("cpu") if device is None else device

        self.latent_dimension = self.config.problem_specifics["latent_dimension_job_seq"]

        # Attention layer which is applied to the situation vector as query
        # and the remaining jobs as keys in order to get a transformed board vector, which is subsequently used
        # for computing the policy.
        self.board_attention = AttentionNarrow(
            query_dim=self.latent_dimension,
            key_dim=self.latent_dimension,
            latent_dim=self.latent_dimension,
            n_attention_heads=self.config.problem_specifics["num_attention_heads"],
            use_query_linear=True
        )

        # Attended board vector gets piped through feedforward before final policy attention
        self.board_feedforward = FeedForward(
            input_dim=self.latent_dimension,
            hidden_dim=4*self.latent_dimension,
            output_dim=self.latent_dimension
        )

        # Attention layer which is applied to the feedforwarded, attended board vector to get the policy.
        self.policy_attention = AttentionNarrow(query_dim=self.latent_dimension,
                                                key_dim=self.latent_dimension,
                                                latent_dim=self.latent_dimension,
                                                n_attention_heads=1,
                                                use_query_linear=False
                                                )

        # Indicator embedding of whether the current board is the second player or not
        self.second_player_embedding = nn.Linear(1, self.latent_dimension, bias=False)

        # Feedforward to get value from board vector (concatenated situations of both players)
        self.value_feedforward = nn.Sequential(
            nn.Linear(2 * self.latent_dimension, 2 * self.latent_dimension),
            nn.GELU(),
            nn.Linear(2 * self.latent_dimension, 2 * self.latent_dimension),
            nn.GELU(),
            nn.Linear(2 * self.latent_dimension, 2 * self.latent_dimension),
            nn.GELU(),
            nn.Linear(2 * self.latent_dimension, 1)
        )

    def forward(self, situation_vector_current, situation_vector_opponent, jobs_transformed,
                policy_attention_mask):
        batch_size = situation_vector_current.shape[0]

        # Concatenate situation vectors of both players to a board vector
        board_vector_value = torch.cat([situation_vector_current, situation_vector_opponent], dim=1)

        # Use the board vector to obtain the values in [-1, 1] via a feedforward
        values = self.value_feedforward(board_vector_value)
        values = torch.tanh(values)

        # Use the board vector as query to perform a multi-headed attention pass w.r.t. remaining nodes, before
        # applying final policy attention
        # The attention weights are padded to the maximum remaining nodes sequence
        board_vector_transformed, _ = self.board_attention(
            query=situation_vector_current,
            keys=jobs_transformed,
            padding_mask=policy_attention_mask
        )

        # Use residual connection here
        board_vector_transformed_ff = self.board_feedforward(board_vector_transformed)
        board_vector_transformed = board_vector_transformed + board_vector_transformed_ff

        # Use the transformed board vector as query to perform a one-headed attention pass and use the attention weights
        # as basis for the policy computation
        # The attention weights are padded to the maximum remaining nodes sequence
        _, attention_weights_unmasked = self.policy_attention(
            query=board_vector_transformed,
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
