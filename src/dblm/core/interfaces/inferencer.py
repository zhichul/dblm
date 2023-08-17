from __future__ import annotations
from dblm.core.interfaces import distribution
import abc

class MarginalInferencer(abc.ABC):
    """Returns a list of marginals."""
    @abc.abstractmethod
    def inference(self, model: distribution.Distribution, observation: dict[int, int], query: list[int]) -> list[distribution.Distribution]:
        ...
