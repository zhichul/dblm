import torch
from dblm.core.interfaces import sampler
from dblm.core.interfaces import pgm

class TabularSampler(sampler.Sampler):

    def sample(self, n, table: pgm.ProbabilityTable):
        probs = table.probability_table()
        num_vars = len(probs.size())
        categorical = torch.distributions.Categorical(probs=probs.reshape(-1))
        samples = categorical.sample(torch.Size([n]))
        sample_slices = [torch.tensor(0)] * num_vars
        for i in range(num_vars):
            num_values = probs.size(-1-i)
            sample_slices[-1-i] = samples.to(torch.int) % num_values
            samples = (samples / num_values).floor()
        return torch.stack(sample_slices, dim=1)
