import torch
from dblm.core.modeling import pgm
import torch.nn as nn


import enum


class TensorInitializer(enum.Enum):
    CONSTANT = enum.auto()
    GAUSSIAN = enum.auto()
    UNIFORM = enum.auto()


class LogProbTable(nn.Module, pgm.ProbabilityTable):

    def __init__(self, size: int, initializer: TensorInitializer, requires_grad:bool=True) -> None:
        super().__init__()
        # Store as ndarray
        if initializer is TensorInitializer.CONSTANT:
            self.logits = nn.Parameter(torch.zeros(size), requires_grad=requires_grad)
        elif initializer is TensorInitializer.UNIFORM:
            self.logits = nn.Parameter(torch.rand(size), requires_grad=requires_grad)
        else:
            raise NotImplementedError()

    def probability_table(self):
        original_size = self.logits.size()
        flattened_logits = self.logits.view(-1)
        return flattened_logits.softmax(dim=-1).view(original_size)


    def log_probability_table(self):
        original_size = self.logits.size()
        flattened_logits = self.logits.view(-1)
        return flattened_logits.log_softmax(dim=-1).view(original_size)

    def likelihood_function(self, assignment):
        return self.probability_table()[assignment]

    def log_likelihood_function(self, assignment):
        return self.log_probability_table()[assignment]
