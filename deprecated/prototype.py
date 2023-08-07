from __future__ import annotations
import abc


from typing import Protocol


class TopologicalSortable(Protocol):
    def topological_order(self) -> list[int]:
        ...

class Distribution:
    """A class that has information about the graph."""
    ...

class GraphDistribution(Distribution):

    def graph(self):
        ...
    
    def directed_graph(self):
        ...

    def topological_order(self):
        ...

    def local_distribution(self):
        ...


class Sampler(abc.ABC):
    @abc.abstractmethod
    def sample(self, distribution: Distribution):
        ...



class TopoSampler(Sampler):

    def sample(self, distribution: TopologicalSortable):
        ...


class Pipeline:
    def __init__(self):
        self.dag = {}

    def add_step(self):
        ...

    def run():
        ...
    
class Step:
    ...

    def accept(input):
        ...


class SamplingStep:
    def __init__(self, sampler):
        self.sampler = sampler


class SaveArtifact:
    def __init__(self, path) -> None:
        self.path = path

    def accept(input):
        """save that to disk!"""


def main():
    pipeline = Pipeline()
    output = pipeline.add_sequential(
        CreateDistributionStep(GraphDistribution()),
        SamplingStep(),
        SaveArtifact(),
    )
    pipeline.add_step(
        inputs=[
            output,
            GraphDistribution()
        ],
        TopoSampler()
    )

    output.materialize()
