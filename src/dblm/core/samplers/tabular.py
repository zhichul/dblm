from dblm.core.samplers import sampler
from dblm.core.modeling import pgm

class TabularSampler(sampler.Sampler):

    def sample(self, distribution: pgm.ProbabilityTable):
        ...
