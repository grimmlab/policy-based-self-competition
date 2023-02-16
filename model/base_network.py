from typing import Dict, List

import torch
from torch import nn

from abc import ABC, abstractmethod
from base_config import BaseConfig


class BaseNetwork(ABC, nn.Module):
    @abstractmethod
    def __init__(self, config: BaseConfig, device: torch.device = None):
        super().__init__()

    @abstractmethod
    def forward(self, x: Dict):
        pass

    @staticmethod
    @abstractmethod
    def states_to_batch(states: List, config: BaseConfig, to_device: torch.device = None):
        pass

    @staticmethod
    def states_batch_dict_to_device(batch_dict: Dict, to_device: torch.device):
        pass

    def set_weights(self, weights):
        if weights is not None:
            self.load_state_dict(weights)

    def get_weights(self):
        return dict_to_cpu(self.state_dict())


class FeedForward(nn.Module):
    """
    Simple MLP Network
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict