import torch
from dblm.core.inferencers import belief_propagation
from dblm.core.interfaces import pgm, sampler


class BatchTreeSampler(sampler.Sampler):

    def __init__(self) -> None:
        super().__init__()
        self.bp = belief_propagation.FactorGraphBeliefPropagation()

    def sample(self, n, distribution: pgm.FactorGraphModel):
        #TODO check that it is actually a tree
        observations = {}
        for i in range(distribution.nvars):
            inference_results = self.bp.inference(distribution, observations, [i], iterations=2 * distribution.nvars)
            table = inference_results.query_marginals[0].log_potential_table()
            if i == 0:
                table = table.expand(n, *table.size())# expand batch dimension
            observation_i = torch.distributions.Categorical(logits=table).sample()
            observations[i] = observation_i
        return torch.stack([observations[i] for i in range(distribution.nvars)], dim=1)
