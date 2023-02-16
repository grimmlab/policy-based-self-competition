import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter


class PaddedSequenceNormalization(nn.Module):

    def __init__(self, embed_dim: int, affine: bool = True, eps: float = 1e-5):
        super(PaddedSequenceNormalization, self).__init__()

        self.embed_dim = embed_dim
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.weight = Parameter(torch.empty((1, 1, embed_dim)))
            self.bias = Parameter(torch.empty((1, 1, embed_dim)))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: Tensor, padding: Tensor):
        """
        Parameters
        ----------
        input: [Tensor] of shape (<batch size>, <sequence length>, <embed dim>)
        padding: [Tensor] of shape (<batch size>, <sequence length>) with 1 for each valid
            sequence item and 0 for each padded one.

        Returns
        -------
            [Tensor] Normalized sequence of same shape as input.
        """
        batch_size = input.shape[0]
        num_seq_items = torch.sum(padding, dim=1)

        # Compute the mean vector of each input sequence individually
        mean_input = torch.sum(input * padding.unsqueeze(-1), dim=1, keepdim=True) / \
                     num_seq_items.view((batch_size, 1, 1))

        output = input - mean_input

        # Compute the variance of each input sequence individually
        variance = torch.sum((output**2) * padding.unsqueeze(-1), dim=1, keepdim=True) / \
                     (num_seq_items - 1).view((batch_size, 1, 1))
        std = torch.sqrt(variance + self.eps)

        output = output / std
        output = output * self.weight + self.bias

        return output


class PaddedBatchNorm1d(nn.BatchNorm1d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None,
                 dtype=None):
        super(PaddedBatchNorm1d, self).__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)

    def forward(self, input: Tensor, padding: Tensor):
        """
        Parameters
        ----------
        input: [Tensor] of shape (<batch size>, <sequence length>, <embed dim>)
        padding: [Tensor] of shape (<batch size>, <sequence length>) with 1 for each valid
            sequence item and 0 for each padded one.

        Returns
        -------
            [Tensor] Normalized sequence of same shape as input.
        """
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            n = torch.count_nonzero(padding)
            # Compute the padded mean
            mean = torch.sum(input * padding[:, :, None], dim=[0, 1]) / n

            var = torch.sum( ((input - mean[None, None, :])**2) * padding[:, :, None], dim=[0, 1] ) / n

            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, None, :]) / (torch.sqrt(var[None, None, :] + self.eps))
        if self.affine:
            input = input * self.weight[None, None, :] + self.bias[None, None, :]

        return input