from dblm.core.interfaces import sampler
from dblm.core.interfaces import pgm

class AncestralSampler(sampler.Sampler):

    def sample(self, distribution: pgm.BayesianNetwork):
        ...
