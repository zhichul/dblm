
from dblm.core.interfaces import featurizer
import torch
import torch.nn as nn

class XTermFrequency(featurizer.Featurizer, nn.Module):

    def __init__(self, vocab_size:int) -> None:
        self.vocab_size = vocab_size

    @property
    def out_features(self):
        return self.vocab_size

    def forward(self, assignments):
        """Assumes that assignments contain only the assignments to x"""
        assert len(assignments.size()) == 2 and assignments.max().item() < self.vocab_size
        counts: torch.Tensor = (assignments[...,None] == torch.arange(self.vocab_size)[None, None, :]).sum(dim=1)
        frequency = counts / counts.sum(dim=-1,keepdim=True)
        return frequency

if __name__ == "__main__":
    extractor = XTermFrequency(10)
    sentences = torch.tensor([
        [-1,0,-1,0,-1,9,-1,9],
        [-1,0,-1,8,-1,9,-1,9],
        ])
    print(sentences)
    print(extractor.forward(sentences))
