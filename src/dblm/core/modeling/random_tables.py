import torch
from dblm.core.modeling import pgm
import torch.nn as nn


import enum


class TensorInitializer(enum.Enum):
    UNIFORM = enum.auto()
    GAUSSIAN = enum.auto()


class LogProbTable(nn.Module, pgm.ProbabilityTable):

    def __init__(self, size: int, intializer: TensorInitializer, requires_grad:bool=True) -> None:
        super().__init__()
        # Store as ndarray
        if intializer is TensorInitializer.UNIFORM:
            self.logprob = nn.Parameter(torch.zeros(size), requires_grad=requires_grad)
        else:
            raise NotImplementedError()

    def probability_table(self):
        original_size = self.logprob.size()
        flattened_logprob = self.logprob.view(-1)
        return flattened_logprob.softmax(dim=-1).view(original_size)
