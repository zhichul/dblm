from dblm.core.samplers import ancestral
from dblm.utils import seeding


def generate_data(n, model, observable_vars, seed=42):
    seeding.seed(seed)
    sampler = ancestral.AncestralSamplerWithPotentialTables()
    data = sampler.sample(n, model)[:,observable_vars] # type:ignore
    return data
