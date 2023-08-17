import abc
import torch.nn as nn

class Featurizer(nn.Module, abc.ABC):

    @property
    @abc.abstractmethod
    def out_features(self):
        raise NotImplementedError()