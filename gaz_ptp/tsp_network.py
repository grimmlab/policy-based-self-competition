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

        self.latent_dimension = self.config.problem_specifics["latent_dimension"]

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
        remaining_nodes_transformed_list = []

        # We fold the opponent board into the batch dimension so that we can compute the situation vectors
        # for both players in one pass.
        batch_size = boards["current"]["start_nodes_batch"].shape[0]
        combined_board = dict()
        combined_board["start_nodes_batch"] = torch.cat(
            (boards["current"]["start_nodes_batch"], boards["opponent"]["start_nodes_batch"]), dim=0)
        combined_board["start_nodes_lookup_embedding_idcs"] = torch.cat((
            boards["current"]["start_nodes_lookup_embedding_idcs"],
            boards["opponent"]["start_nodes_lookup_embedding_idcs"] + batch_size  # shift indices by batch_size
        ))
        combined_board["end_nodes_batch"] = torch.cat(
            (boards["current"]["end_nodes_batch"], boards["opponent"]["end_nodes_batch"]), dim=0)
        combined_board["end_nodes_lookup_embedding_idcs"] = torch.cat((
            boards["current"]["end_nodes_lookup_embedding_idcs"],
            boards["opponent"]["end_nodes_lookup_embedding_idcs"] + batch_size  # shift indices by batch_size
        ))
        combined_board["remaining_nodes_batch"] = torch.cat(
            (boards["current"]["remaining_nodes_batch"], boards["opponent"]["remaining_nodes_batch"]), dim=0)
        combined_board["tour_length_batch"] = torch.cat(
            (boards["current"]["tour_length_batch"], boards["opponent"]["tour_length_batch"]), dim=0)
        combined_board["num_remaining_nodes_embedding_batch"] = torch.cat(
            (boards["current"]["num_remaining_nodes_embedding_batch"],
             boards["opponent"]["num_remaining_nodes_embedding_batch"]), dim=0)
        combined_board["self_attention_masks_batch"] = torch.cat(
            (boards["current"]["self_attention_masks_batch"],
             boards["opponent"]["self_attention_masks_batch"]), dim=0)

        combined_board["distance_matrix"] = torch.cat(
            (boards["current"]["distance_matrix"], boards["opponent"]["distance_matrix"]), dim=0)

        situation_vector_batch, remaining_nodes_transformed_batch = self.situation_net(combined_board)
        situation_vectors.append(situation_vector_batch[:batch_size])  # for player 1
        situation_vectors.append(situation_vector_batch[batch_size:])  # for player -1
        remaining_nodes_transformed_list.append(remaining_nodes_transformed_batch[:batch_size])
        remaining_nodes_transformed_list.append(remaining_nodes_transformed_batch[batch_size:])

        policy_logits_padded_masked, values = self.prediction_net(
            situation_vector_current=situation_vectors[0],
            situation_vector_opponent=situation_vectors[1],
            remaining_nodes=remaining_nodes_transformed_list[0],
            policy_attention_mask=boards["current"]["policy_attention_masks_batch"]
        )

        return policy_logits_padded_masked, values

    @staticmethod
    def states_to_batch(states: List, config: BaseConfig, to_device: torch.device = None):
        """
        Given a list of canonical boards, each of which is itself a 2-element list with the player's situation as
        returned from `get_canonical_board`, prepares all the tensors and index lists needed for a batched forward pass.
        For historical reasons we denote by `current` player 1, and by `opponent` player 2.
        """
        # Get the maximum sequence length of all remaining nodes sequences, as we are later batching current and
        # opponent situations for faster forward pass.
        boards = states
        max_seq_len = max([max(board[0]["remaining_nodes"].shape[0], board[1]["remaining_nodes"].shape[0])
                           for board in boards])

        return {
            "current": Network.player_situations_to_batch(
                situations=[board[0] for board in boards], config=config, to_device=to_device,
                custom_max_seq_len=max_seq_len
            ),
            "opponent": Network.player_situations_to_batch(
                situations=[board[1] for board in boards], config=config, to_device=to_device,
                custom_max_seq_len=max_seq_len
            )
        }

    @staticmethod
    def player_situations_to_batch(situations: List[Dict], config: BaseConfig, to_device: torch.device = None,
                                   custom_max_seq_len: int = 0):
        """
        Parameters
        ----------
        situations
        config
        to_device
        custom_max_seq_len: [int] Defaults to 0. If given, this is used as the maximum sequence length, and not
            the maximum of `sequence_lens`
        """
        if not to_device:
            to_device = torch.device("cpu")

        batch_size = len(situations)

        # Prepare batch of start_node tensors. If start_node is None, we give a (0, 0) node, which is later
        # replaced with the lookup embedding
        start_nodes_batch = torch.stack(
            [
                situation["start_node"] if situation["start_node"] is not None else torch.tensor([0., 0.])
                for situation in situations
            ]
        ).float().to(to_device)
        # Make a long tensor with indices of start nodes which need to be replaced by lookup embedding
        # (as they have been None)
        start_nodes_lookup_embedding = torch.LongTensor([i for i, situation in enumerate(situations)
                                                         if situation["start_node"] is None]).to(to_device)

        # Prepare batch of end_node tensors. If end_node is None, we give a (0, 0) node, which is later
        # replaced with the lookup embedding
        end_nodes_batch = torch.stack(
            [
                situation["end_node"] if situation["end_node"] is not None else torch.tensor([0., 0.])
                for situation in situations
            ]
        ).float().to(to_device)
        # Make a long tensor with indices of end nodes which need to be replaced by lookup embedding
        # (as they have been None)
        end_nodes_lookup_embedding = torch.LongTensor([i for i, situation in enumerate(situations)
                                                       if situation["end_node"] is None]).to(to_device)

        # We now prepare the batch of remaining nodes, which builds the heart of the graph. For this we need
        # to pad the sequence of remaining nodes to the same length, so that batching works
        remaining_nodes_sequence_lens = [situation["remaining_nodes"].shape[0] for situation in situations]
        max_seq_len = max(remaining_nodes_sequence_lens) if not custom_max_seq_len else custom_max_seq_len
        remaining_nodes_sequence_lens_batch = torch.LongTensor(remaining_nodes_sequence_lens).to(to_device)
        # float version for embedding the number of remaining nodes
        num_remaining_nodes_embedding_batch = torch.tensor(remaining_nodes_sequence_lens).float().unsqueeze(-1).to(
            to_device)

        remaining_nodes_batch = torch.stack([
            nn.functional.pad(situation["remaining_nodes"],
                              pad=(0, 0, 0, max_seq_len - remaining_nodes_sequence_lens[i]))
            for i, situation in enumerate(situations)
        ]).float().to(to_device)

        self_attention_masks_batch = torch.ones((batch_size, max_seq_len + 5))
        policy_attention_masks_batch = torch.ones((batch_size, max_seq_len))
        for i, seq_len in enumerate(remaining_nodes_sequence_lens):
            self_attention_masks_batch[i, seq_len + 5:] = 0
            policy_attention_masks_batch[i, seq_len:] = 0
        self_attention_masks_batch = self_attention_masks_batch.float().to(to_device)
        policy_attention_masks_batch = policy_attention_masks_batch.float().to(to_device)

        # mask which is applied to attentions, i.e. masking tensors of size (batch, heads, queries, seq_len)
        self_attention_masks_batch = self_attention_masks_batch.view(batch_size, 1, 1, max_seq_len + 5)
        # Create the attention mask when we are performing attention with board vector as query and keys
        # the remaining nodes (in order to decide which node to choose next). Note that for this
        # attention step we only attend to the remaining nodes
        policy_attention_masks_batch = policy_attention_masks_batch.view(batch_size, 1, 1, max_seq_len)

        # Batch the current tour lengths
        tour_length_batch = torch.tensor([situation["tour_length"] for situation in situations]) \
            .float().unsqueeze(-1).to(to_device)

        # Pad the distance matrix of remaining nodes to the maximum sequence length to the right (down) and
        # number of global tokens to the left (up) (their distances are learnable embeddings)
        distance_matrix = torch.stack([
            nn.functional.pad(situation["remaining_nodes_distance_matrix"],
                              pad=(5, max_seq_len - remaining_nodes_sequence_lens[i],
                                   5, max_seq_len - remaining_nodes_sequence_lens[i]))
            for i, situation in enumerate(situations)
        ])

        # In cases where the start/end nodes are set, we place them within the distance matrix
        for i, situation in enumerate(situations):
            start_end_node_distance_matrix = situation["start_end_node_distance_matrix"]
            if start_end_node_distance_matrix is None:
                continue
            l = start_end_node_distance_matrix.shape[1]
            distance_matrix[i, 3:5, 5:5 + l] = start_end_node_distance_matrix
            distance_matrix[i, 5:5 + l, 3:5] = torch.transpose(start_end_node_distance_matrix, 0, 1)

        distance_matrix = distance_matrix.float().to(to_device)

        return {
            "start_nodes_batch": start_nodes_batch,  # shape (batch_size, 2)
            "start_nodes_lookup_embedding_idcs": start_nodes_lookup_embedding,
            # shape (<num of instances where start node is None>)
            "end_nodes_batch": end_nodes_batch,  # shape (batch_size, 2)
            "end_nodes_lookup_embedding_idcs": end_nodes_lookup_embedding,
            # shape (<num of instances where end node is None>)
            "remaining_nodes_sequence_lengths": remaining_nodes_sequence_lens_batch,  # shape (batch_size,)
            "remaining_nodes_sequence_lens_list": remaining_nodes_sequence_lens,  # same as above, only as list
            "num_remaining_nodes_embedding_batch": num_remaining_nodes_embedding_batch,  # shape (batch_size, 1)
            "remaining_nodes_batch": remaining_nodes_batch,  # shape (batch_size, max_sequence_len, 2)
            "self_attention_masks_batch": self_attention_masks_batch,  # shape (batch_size, 1, 1, max_seq_len + 5)
            "policy_attention_masks_batch": policy_attention_masks_batch,  # shape (batch_size, 1, 1, max seq len)
            "tour_length_batch": tour_length_batch,  # shape (batch_size, 1)
            "distance_matrix": distance_matrix,
            # shape (batch_size, pad_remaining_nodes_to + num_global_tokens, max_sequence_len + 5)
            "policy_padded_to": max_seq_len,  # int
        }

    @staticmethod
    def states_batch_dict_to_device(batch_dict: Dict, to_device: torch.device):
        for i in batch_dict:
            for key in batch_dict[i]:
                if key not in ["remaining_nodes_sequence_lens_list", "policy_padded_to"]:
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

        self.latent_dimension = self.config.problem_specifics["latent_dimension"]
        self.num_heads = self.config.problem_specifics["num_attention_heads"]

        # Affine Embedding of 2-dimensional nodes into latent space
        self.node_embedding_linear = nn.Linear(2, self.latent_dimension, bias=True)

        # Affine Embedding of "current tour length" and "number of nodes" into latent space
        self.tour_length_embedding_linear = nn.Linear(1, self.latent_dimension, bias=True)
        self.num_nodes_embedding_linear = nn.Linear(1, self.latent_dimension, bias=True)

        # Lookup embedding of start/end-nodes when not specified. Start Node: 0, end node: 1
        self.start_end_node_lookup = nn.Embedding(num_embeddings=2, embedding_dim=self.latent_dimension)

        # Lookup embedding of indicators of start/end-nodes (get added to the node embedding to guide network that
        # these are start/end nodes). Start node indicator: 0, end node indicator: 1
        self.start_end_node_indicator_lookup = nn.Embedding(num_embeddings=2, embedding_dim=self.latent_dimension)

        # The situation token is an additional element in the sequence which gets piped through the transformer
        # The transformed token is used as a latent representation of the situation. This is comparable to a <cls>-token
        # in NLP
        self.situation_token_lookup = nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dimension)

        # Affine transformation for the distance matrix which gets later added to the attention weights
        self.distance_matrix_linear = nn.Linear(in_features=1, out_features=self.num_heads)

        # Lookup embedding of the distance of start/end-node to other nodes when start/end node is not specified.
        self.distance_start_end_node_lookup = nn.Embedding(num_embeddings=2, embedding_dim=self.num_heads)

        # Lookup embedding for the distance of other global tokens to the other nodes.
        # 0: global graph token, 1: tour length token, 2: num remaining nodes token
        self.distance_global_token_lookup = nn.Embedding(num_embeddings=3, embedding_dim=self.num_heads)

        # The transformer itself for getting the graph representation
        transformer_blocks = []
        for _ in range(self.config.problem_specifics["num_transformer_blocks"]):
            transformer_blocks.append(
                 TransformerBlock(
                     latent_dim=self.latent_dimension,
                     n_attention_heads=self.num_heads,
                     feedforward_hidden_dim_mult=4,
                     dropout=0.0,
                     normalization="layer",
                     use_mask=True,
                     use_attention_bias=True
                 )
            )
        self.graph_transformer = nn.ModuleList(transformer_blocks)

    def forward(self, board_situation):
        """
        Parameters
        ----------
        board_situation: [Dict] Situation batch dictionary as
            returned from `TSPTransformerNetwork.player_situations_to_batch`
        Returns
        -------
            [torch.Tensor] Situation vector as tensor of shape (batch, latent_dim)
            [torch.Tensor] Transformed remaining nodes as tensor of shape (batch, <max num remaining nodes in batch>, latent_dim)
        """

        # Embed start and end nodes. Replace with lookup embedding if necessary
        batch_size = board_situation["start_nodes_batch"].shape[0]
        start_nodes_embedding = self.node_embedding_linear(board_situation["start_nodes_batch"])
        lookup_idcs = board_situation["start_nodes_lookup_embedding_idcs"]
        num_lookups = lookup_idcs.size(dim=0)
        if num_lookups > 0:
            idcs = torch.LongTensor([0] * num_lookups).to(self.device)
            lookup_start_nodes = self.start_end_node_lookup(idcs)
            # replace the embedded start nodes in the batch with the lookups where necessary
            start_nodes_embedding[lookup_idcs] = lookup_start_nodes

        # Add the start node indicator lookup to the start_nodes_embedding
        start_node_indicator = self.start_end_node_indicator_lookup(
            torch.LongTensor([0] * batch_size).to(self.device)
        )
        start_nodes_embedding = start_nodes_embedding + start_node_indicator

        # Do the same for end nodes
        end_nodes_embedding = self.node_embedding_linear(board_situation["end_nodes_batch"])
        lookup_idcs = board_situation["end_nodes_lookup_embedding_idcs"]
        num_lookups = lookup_idcs.size(dim=0)
        if num_lookups > 0:
            idcs = torch.LongTensor([1] * num_lookups).to(self.device)
            lookup_end_nodes = self.start_end_node_lookup(idcs)
            # replace the embedded start nodes in the batch with the lookups where necessary
            end_nodes_embedding[lookup_idcs] = lookup_end_nodes

        # Add the start node indicator lookup to the start_nodes_embedding
        end_node_indicator = self.start_end_node_indicator_lookup(
            torch.LongTensor([1] * batch_size).to(self.device)
        )
        end_nodes_embedding = end_nodes_embedding + end_node_indicator

        # Embed the remaining nodes
        remaining_nodes_embedding = self.node_embedding_linear(board_situation["remaining_nodes_batch"])

        # Embed the tour length and number of remaining nodes
        tour_length_embedding = self.tour_length_embedding_linear(board_situation["tour_length_batch"])
        num_remaining_nodes_embedding = self.num_nodes_embedding_linear(board_situation["num_remaining_nodes_embedding_batch"])

        # Get the situation token which will result in our latent situation
        situation_token = self.situation_token_lookup(
            torch.LongTensor([0] * batch_size).to(self.device)
        )

        # Get the attention bias for all blocks given by the distance matrix. We do this in multiple steps.
        # Step 1: Make affine transformation of distance matrix of remaining nodes
        distance_matrix = board_situation["distance_matrix"]  # (b, n, n)
        n = distance_matrix.shape[1]
        distance_matrix = distance_matrix.view(batch_size, -1, 1)  # (b, n*n, 1)
        attention_bias_matrix = self.distance_matrix_linear(distance_matrix).view(
            batch_size, n, n, self.num_heads)  # (b, n, n, num_heads)

        # Set the lookup distances for start and end nodes in the cases where lookup is necessary
        if num_lookups > 0:
            start_end_distance = self.distance_start_end_node_lookup.weight.view(1, 2, 1, self.num_heads)
            start_end_distance = start_end_distance.repeat(num_lookups, 1, n - 5, 1)
            # ==> start_end_distance [num_lookups, 2, num_remaining_nodes, num_heads)
            attention_bias_matrix[lookup_idcs, 3:5, 5:, :] = start_end_distance
            attention_bias_matrix[lookup_idcs, 5:, 3:5, :] = start_end_distance.transpose(1, 2)

        # Now set the lookup distances for the global tokens
        for i in range(3):
            global_token_distance_lookup = self.distance_global_token_lookup.weight[i:i+1].view(1, 1, 1, self.num_heads)
            global_token_distance_lookup = global_token_distance_lookup.repeat(batch_size, 1, n - (i + 1), 1)
            attention_bias_matrix[:, i:i+1, i+1:, :] = global_token_distance_lookup
            attention_bias_matrix[:, i + 1:, i:i + 1, :] = global_token_distance_lookup.transpose(1, 2)

        # Stack global tokens, start and end nodes, tour length and number of remaining nodes on
        # top of the remaining nodes embedding to get the full graph
        graph_nodes_embedding = torch.cat([situation_token.unsqueeze(1),
                                           tour_length_embedding.unsqueeze(1),
                                           num_remaining_nodes_embedding.unsqueeze(1),
                                           start_nodes_embedding.unsqueeze(1),
                                           end_nodes_embedding.unsqueeze(1),
                                           remaining_nodes_embedding
                                           ],
                                          dim=1)

        # Attention bias matrix is of shape (batch, seq len, seq len, num heads)
        # We need to transform it to shape (batch * num heads, sequence_len, sequence_len)
        attention_bias_matrix = attention_bias_matrix.view(batch_size, n * n, self.num_heads)
        attention_bias_matrix = attention_bias_matrix.transpose(1, 2).view(batch_size, self.num_heads, n, n).contiguous()

        # The graph nodes now get piped through the transformer
        graph_nodes_transformed = graph_nodes_embedding
        padding_mask = board_situation["self_attention_masks_batch"]
        for trf_block in self.graph_transformer:
            graph_nodes_transformed = trf_block(graph_nodes_transformed,
                                                padding_mask=padding_mask, attention_bias=attention_bias_matrix)

        # We now have the transformed embeddings of the graph nodes.
        # Slice out transformed situation token (which is the first global token) and remaining nodes
        situation_token_transformed = graph_nodes_transformed[:, 0, :]
        remaining_nodes_transformed = graph_nodes_transformed[:, 5:, :]

        return situation_token_transformed, remaining_nodes_transformed


class PredictionNetwork(torch.nn.Module):
    def __init__(self, config: BaseConfig, device: torch.device = None):
        super().__init__()

        self.config = config
        self.device = torch.device("cpu") if device is None else device

        self.latent_dimension = self.config.problem_specifics["latent_dimension"]

        # Attention layer which is applied to the board vector (concatenation of both situation vectors) as query
        # and the remaining nodes as keys in order to get a transformed board vector, which is subsequently used
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

        # Feedforward to get value from board vector (concatenated situations of both players)
        self.value_feedforward = nn.Sequential(
            nn.Linear(2 * self.latent_dimension, self.latent_dimension),
            nn.GELU(),
            nn.Linear(self.latent_dimension, self.latent_dimension),
            nn.GELU(),
            nn.Linear(self.latent_dimension, 1)
        )

    def forward(self, situation_vector_current, situation_vector_opponent, remaining_nodes,
                policy_attention_mask):
        batch_size = situation_vector_current.shape[0]
        # Concatenate situation vectors of both players to a board vector
        board_vector = torch.cat([situation_vector_current, situation_vector_opponent], dim=1)

        # Use the board vector to obtain the values in [-1, 1] via a feedforward
        values = self.value_feedforward(board_vector)
        values = torch.tanh(values)

        # Use the board vector as query to perform a multi-headed attention pass w.r.t. remaining nodes, before
        # applying final policy attention
        # The attention weights are padded to the maximum remaining nodes sequence
        board_vector_transformed, _ = self.board_attention(
            query=situation_vector_current,
            keys=remaining_nodes,
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
            keys=remaining_nodes,
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


