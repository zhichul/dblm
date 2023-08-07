from dblm.core.modeling import distribution
import abc


class Sampler(abc.ABC):

    @abc.abstractmethod
    def sample(self, distribution: distribution.Distribution):
        ...
