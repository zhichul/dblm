from dblm.core.interfaces import distribution
import abc


class Sampler(abc.ABC):

    @abc.abstractmethod
    def sample(self, distribution: distribution.Distribution):
        ...
