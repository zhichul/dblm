from dblm.core.samplers import sampler
from dblm.core.modeling import pgm

class AncestralSampler(sampler.Sampler):

    def sample(self, distribution: pgm.BayesianNetwork):
        ...
