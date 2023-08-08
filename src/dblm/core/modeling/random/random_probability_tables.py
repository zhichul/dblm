from typing import Sequence, Union
from dblm.core.modeling import pgm
from dblm.core.modeling.random import constants
import torch.nn as nn
import torch


class LogLinearTable(nn.Module, pgm.ProbabilityTable, pgm.LocalPotentialFunction):

    def __init__(self, size, initializer: constants.TensorInitializer, requires_grad:bool=True) -> None:
        """If initializer is a tensor, requires_grad is ignored."""
        super().__init__()
        # Store as ndarray
        if initializer is constants.TensorInitializer.CONSTANT:
            self.logits = nn.Parameter(torch.zeros(size), requires_grad=requires_grad)
        elif initializer is constants.TensorInitializer.UNIFORM:
            self.logits = nn.Parameter(torch.rand(size), requires_grad=requires_grad)
        else:
            raise NotImplementedError()

    def probability_table(self) -> torch.Tensor:
        original_size = self.logits.size()
        flattened_logits = self.logits.reshape(-1)
        return flattened_logits.softmax(dim=-1).reshape(original_size)

    def potential_table(self) -> torch.Tensor:
        return self.logits.exp()

    def log_probability_table(self) -> torch.Tensor:
        original_size = self.logits.size()
        flattened_logits = self.logits.reshape(-1)
        return flattened_logits.log_softmax(dim=-1).reshape(original_size)

    def log_potential_table(self) -> torch.Tensor:
        return self.logits.clone()

    def likelihood_function(self, assignment):
        return self.probability_table()[assignment]

    def log_likelihood_function(self, assignment):
        return self.log_probability_table()[assignment]

    @staticmethod
    def from_logits(logits):
        table = LogLinearTable(logits.size(), constants.TensorInitializer.CONSTANT)
        table.logits = logits
        return table
