from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
import math
from model.normalization import PaddedSequenceNormalization, PaddedBatchNorm1d


class TransformerBlock(nn.Module):
    """
    Transformer block with pre-norm and feature narrow self-attention.
    Based on the wonderful blog post https://peterbloem.nl/blog/transformers
    """
    def __init__(self, latent_dim, n_attention_heads, feedforward_hidden_dim_mult=2, dropout=0.0, normalization="instance",
                 use_mask: bool = True, use_attention_bias: bool = False):
        super().__init__()

        self.use_mask = use_mask
        self.use_attention_bias = use_attention_bias
        self.attention = SelfAttentionNarrow(latent_dim=latent_dim, n_attention_heads=n_attention_heads,
                                             use_mask=use_mask, use_attention_bias=use_attention_bias)

        self.norm_type = normalization
        # normalization before the self-attention
        if normalization == "layer":
            self.norm1 = nn.LayerNorm(latent_dim)
            self.norm2 = nn.LayerNorm(latent_dim)
        elif normalization == "sequence":
            self.norm1 = PaddedSequenceNormalization(latent_dim)
            self.norm2 = PaddedSequenceNormalization(latent_dim)
        elif normalization == "batch":
            self.norm1 = PaddedBatchNorm1d(latent_dim)
            self.norm2 = PaddedBatchNorm1d(latent_dim)
        else:
            raise Exception("Normalization type unknown")

        self.feedforward = nn.Sequential(
            nn.Linear(latent_dim, feedforward_hidden_dim_mult * latent_dim),
            nn.GELU(),
            nn.Linear(feedforward_hidden_dim_mult * latent_dim, latent_dim)
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, padding_mask: Optional[torch.Tensor] = None,
                attention_bias: Optional[torch.Tensor] = None):
        # attention_bias is expected in shape (batch * num_heads, sequence_len, sequence_len)

        if self.norm_type in ["sequence", "batch"]:
            x_normed = self.norm1(x, padding_mask.view(padding_mask.shape[0], padding_mask.shape[-1]))
        else:
            x_normed = self.norm1(x)
        attention = self.attention(x_normed, padding_mask=padding_mask, attention_bias=attention_bias)
        x = attention + x

        if self.norm_type in ["sequence", "batch"]:
            x_mlp = self.norm2(x, padding_mask.view(padding_mask.shape[0], padding_mask.shape[-1]))
        else:
            x_mlp = self.norm2(x)

        x_mlp = self.dropout(x_mlp)
        x_mlp = self.feedforward(x_mlp)

        x = x_mlp + x

        x = self.dropout(x)

        return x


class SelfAttentionNarrow(nn.Module):
    """
    Multi-head self attention module
    With narrow attention, i.e. we split the latent space into smaller
    subspaces and each head performs attention on its own subspace
    """
    def __init__(self, latent_dim, n_attention_heads, use_mask: bool = True, use_attention_bias: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_attention_heads = n_attention_heads

        if self.latent_dim % self.n_attention_heads != 0:
            raise ValueError("Latent dimension should be divisible by number of attention heads")

        self.split_dim = self.latent_dim // self.n_attention_heads

        # Compute keys, queries and values for all heads
        # KQVs are computed for each head on the full vector, and then transformed
        # into (batch_size, sequence_length, n_attention_heads, split_dim)
        self.to_keys_linear = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.to_queries_linear = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.to_values_linear = nn.Linear(self.latent_dim, self.latent_dim, bias=False)

        self.unify_heads_linear = nn.Linear(self.latent_dim, self.latent_dim)
        self.use_mask = use_mask
        self.use_attention_bias = use_attention_bias

    def forward(self, x, padding_mask: Optional[torch.Tensor] = None, attention_bias: Optional[torch.Tensor] = None):
        """
        `padding_mask` is a torch tensor of shape (batch, 1, 1, sequence_len) of ones with 0 for padding tokens

        `attention_bias` gets added to the matrix of attention weights before softmax and is expected as a tensor
        of shape (batch, num_heads, sequence_len, sequence_len)
        """
        attn_mask_penalty = -10000.0

        batch_size, sequence_length, latent_dim = x.shape
        split_dim = self.split_dim
        n_attention_heads = self.n_attention_heads
        values, dot = self.values_and_dotproduct(x)

        # Add attention bias if given
        if self.use_attention_bias:
            dot = dot + attention_bias

        # Before computing softmax, we need to set (dot)_ij = -inf for all i and all j > (masking index of sequence)
        if self.use_mask:
            dot += (1. - padding_mask) * attn_mask_penalty

        # get row-wise self-attention probabilities
        dot = F.softmax(dot, dim=-1)
        dot = dot.view(batch_size * n_attention_heads, sequence_length, sequence_length)

        # apply the self-attention now to the values
        out = torch.bmm(dot, values)  # this has shape (batch_size * n_attention_heads, sequence_length, split_dim)
        out = out.view(batch_size, n_attention_heads, sequence_length, split_dim)

        # transpose back to shape (b, s, h * dim) and unify the heads
        out = out.transpose(1, 2).contiguous().view(batch_size, sequence_length, n_attention_heads * split_dim)

        return self.unify_heads_linear(out)

    def values_and_dotproduct(self, x):
        batch_size, sequence_length, latent_dim = x.shape
        split_dim: int = self.split_dim
        n_attention_heads: int = self.n_attention_heads

        # -- Compute queries, keys and values for attention
        # reshape the output of each linear layer to shape (batch_size, sequence_length, n_attention_heads, split_dim)
        queries = self.to_queries_linear(x).view(batch_size, sequence_length, n_attention_heads, split_dim)
        keys = self.to_keys_linear(x).view(batch_size, sequence_length, n_attention_heads, split_dim)
        values = self.to_values_linear(x).view(batch_size, sequence_length, n_attention_heads, split_dim)

        # -- Compute scaled dot products
        # fold the heads into batch dimension so we can use batch matrix multiplication later
        keys = keys.transpose(1, 2).contiguous().view(batch_size * n_attention_heads, sequence_length, split_dim)
        queries = queries.transpose(1, 2).contiguous().view(batch_size * n_attention_heads, sequence_length, split_dim)
        values = values.transpose(1, 2).contiguous().view(batch_size * n_attention_heads, sequence_length, split_dim)

        # We now can compute the dot product of keys and queries, and scale them
        dot = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(split_dim)

        # in dot, we have (dot)_ij = <query_i, key_j>, so i-th row contains signals of i-th query
        # dot is of shape (batch_size*heads, sequence_length, sequence_length)
        # assert dot.size() == (batch_size * n_attention_heads, sequence_length, sequence_length)

        dot = dot.view(batch_size, n_attention_heads, sequence_length, sequence_length)
        return values, dot


class AttentionNarrow(nn.Module):
    """
    Multi-head attention module, performing attention on a single query.
    With narrow attention, i.e. we split the latent space into smaller
    subspaces and each head performs attention on its own subspace
    """
    def __init__(self, query_dim: int, key_dim: int, latent_dim: int,
                 n_attention_heads: int, use_query_linear: bool = True):
        super().__init__()
        self.query_dim = query_dim  # Dimension of query vector. May be different from latent_dim, and will be projected
                                    # to latent dim
        self.key_dim = key_dim
        self.latent_dim = latent_dim
        self.n_attention_heads = n_attention_heads
        self.use_query_linear = use_query_linear

        if self.latent_dim % self.n_attention_heads != 0:
            raise ValueError("Latent dimension should be divisible by number of attention heads")

        self.split_dim = self.latent_dim // self.n_attention_heads

        # Compute keys, queries and values for all heads
        # KQVs are computed for each head on the full vector, and then transformed
        # into (batch_size, sequence_length, n_attention_heads, split_dim)
        self.to_keys_linear = nn.Linear(self.key_dim, self.latent_dim, bias=False)
        self.to_queries_linear = nn.Linear(self.query_dim, self.latent_dim, bias=False) if use_query_linear else None
        self.to_values_linear = nn.Linear(self.key_dim, self.latent_dim, bias=False)

        self.unify_heads_linear = nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, query, keys, padding_mask):
        """
        Parameters
        ----------
        query: [torch.Tensor] Batched queries of shape (batch, query_dim)
        keys: [torch.Tensor] Batched key sequence of shape (batch, sequence_len, key_dim)
        padding_mask: [torch.Tensor] `padding_mask` is a torch tensor of shape (batch, 1, 1, sequence_len)

        Returns:
            transformed_query: [torch.Tensor] The attention weighted sum of values of shape (batch_size, latent_dim)
            unmasked_attention_weights: [torch.Tensor] Non-softmaxed attention weights of shape (batch_size * num_heads, 1, sequence_len)
                Attention weights of padding elements in sequence are NOT masked!
        """
        attn_mask_penalty = -10000.0

        _keys = keys
        batch_size, sequence_length, _ = keys.shape
        split_dim = self.split_dim
        n_attention_heads = self.n_attention_heads

        # -- Compute queries, keys and values for attention
        # reshape the output of each linear layer to shape (batch_size, sequence_length, n_attention_heads, split_dim)

        if self.use_query_linear:
            query = self.to_queries_linear(query)
        query = query.view(batch_size, 1, n_attention_heads, split_dim)
        keys = self.to_keys_linear(_keys).view(batch_size, sequence_length, n_attention_heads, split_dim)
        values = self.to_values_linear(_keys).view(batch_size, sequence_length, n_attention_heads, split_dim)

        # -- Compute scaled dot products
        # fold the heads into batch dimension so we can use batch matrix multiplication later
        keys = keys.transpose(1, 2).contiguous().view(batch_size * n_attention_heads, sequence_length, split_dim)
        query = query.transpose(1, 2).contiguous().view(batch_size * n_attention_heads, 1, split_dim)
        values = values.transpose(1, 2).contiguous().view(batch_size * n_attention_heads, sequence_length, split_dim)

        # We now can compute the dot product of keys and queries, and scale them
        dot = torch.bmm(query, keys.transpose(1, 2)) / math.sqrt(split_dim)

        # in dot, we have (dot)_ij = <query_i, key_j>, so i-th row contains signals of i-th query
        # dot is of shape (batch_size*heads, 1, sequence_length)
        # assert dot.size() == (batch_size * n_attention_heads, 1, sequence_length)

        dot = dot.view(batch_size, n_attention_heads, 1, sequence_length)

        # Cache attention weights here before softmax to return them
        unmasked_attention_weights_to_return = dot

        # Before computing softmax, we need to set (dot)_ij = -inf for all i and all j > (masking index of sequence)
        dot += (1. - padding_mask) * attn_mask_penalty

        # get row-wise self-attention probabilities
        dot = F.softmax(dot, dim=-1)
        dot = dot.view(batch_size * n_attention_heads, 1, sequence_length)

        # apply the self-attention now to the values
        out = torch.bmm(dot, values)  # this has shape (batch_size * n_attention_heads, 1, split_dim)
        out = out.view(batch_size, n_attention_heads, 1, split_dim)

        # transpose back to shape (b, 1, h * dim) and unify the heads
        out = out.transpose(1, 2).contiguous().view(batch_size, 1, n_attention_heads * split_dim)
        unified = self.unify_heads_linear(out).view(batch_size, self.latent_dim)

        return unified, unmasked_attention_weights_to_return

