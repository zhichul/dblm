from __future__ import annotations
from dblm.core.interfaces import distribution
import abc

class Inferencer(abc.ABC):

    @abc.abstractmethod
    def inference(self, model: distribution.Distribution, observed_vars: list[int], observed_vals: list[int], query_vars: list[int]):
        ...
