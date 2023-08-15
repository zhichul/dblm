from dblm.core.interfaces import sampler
from dblm.core.interfaces import distribution

class GibbsSampler(sampler.Sampler):

    def sample(self, distribution: distribution.Distribution):
        ...
