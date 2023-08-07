from dblm.core.samplers import sampler
from dblm.core.modeling import distribution

class GibbsSampler(sampler.Sampler):

    def sample(self, distribution: distribution.Distribution):
        ...
