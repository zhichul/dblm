from dblm.core.interfaces import distribution
import abc


class Sampler(abc.ABC):

    @abc.abstractmethod
    def sample(self, n, distribution: distribution.Distribution):
        ...
