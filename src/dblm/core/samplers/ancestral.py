from dblm.core.interfaces import sampler
from dblm.core.interfaces import pgm
from dblm.core.samplers import tabular
import torch

class AncestralSamplerWithPotentialTables(sampler.Sampler):

    def __init__(self) -> None:
        super().__init__()
        self.tabular_sampler = tabular.TabularSampler()

    def check_is_joint(self, distribution: pgm.BayesianNetwork):
        local_children = distribution.local_children()
        seen = set()
        for children in local_children:
            for child in children:
                if child in seen:
                    return False
                else:
                    seen.add(child)
        return len(seen) == distribution.nvars

    def sample(self, n, distribution: pgm.BayesianNetwork):
        if not self.check_is_joint(distribution):
            raise ValueError("this sampler requires a joint distribution to be specified")
        # TODO alternatively take an argument that specifies how to sample each individual distribution so that it don't have to be a potential table
        local_factors = distribution.local_distributions()
        local_variables = distribution.local_variables()
        local_parents = distribution.local_parents()
        local_children = distribution.local_children()

        if any(not isinstance(local_factor, pgm.PotentialTable) for local_factor in local_factors):
            raise NotImplementedError("AncestralSamplerWithPotentialTables requires all local distributions implement pgm.PotentialTable.")
        assignments = []
        for _ in range(n):
            assignment = [None] * distribution.nvars
            for i in distribution.topological_order():
                parents = local_parents[i]
                children = local_children[i]
                variables = local_variables[i]
                factor: pgm.PotentialTable = local_factors[i] # type: ignore
                cpt = factor.condition_on({variables.index(p): assignment[p] for p in parents}).renormalize() # type:ignore
                sample = self.tabular_sampler.sample(1, cpt)
                assert sample.nelement() == len(children)
                for child, value in zip(children, sample.reshape(-1).tolist()):
                    assignment[child] = value
            assignments.append(assignment)
        return torch.tensor(assignments)
