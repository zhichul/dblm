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
    # pipeline = Pipeline()
    # output = pipeline.add_sequential(
    #     CreateDistributionStep(GraphDistribution()),
    #     SamplingStep(),
    #     SaveArtifact(),
    # )
    # pipeline.add_step(
    #     inputs=[
    #         output,
    #         GraphDistribution()
    #     ],
    #     TopoSampler()
    # )

    # output.materialize()
    pass