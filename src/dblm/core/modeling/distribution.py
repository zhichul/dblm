import abc


class Distribution(abc.ABC):
    """A class that has information about the graph."""

    @abc.abstractmethod
    def likelihood_function(self, assignment):
        raise NotImplementedError()

    @abc.abstractmethod
    def log_likelihood_function(self, assigment):
        raise NotImplementedError()
