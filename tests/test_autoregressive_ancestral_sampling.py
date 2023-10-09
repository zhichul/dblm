
from dblm.core.modeling import bayesian_networks
from dblm.core.modeling.interleaved_models import autoregressive_models
from dblm.utils import seeding


if __name__ == "__main__":
    import transformers.models.gpt2.modeling_gpt2 as modeling_gpt2
    from dblm.core.modeling import markov_networks
    from dblm.core.modeling import constants
    from dblm.core.samplers import ancestral
    seeding.seed(42)
    transformer = modeling_gpt2.GPT2LMHeadModel(modeling_gpt2.GPT2Config(vocab_size=12 + 3 + 1, n_positions=3 + 1 + 10, bos_token_id=16, n_layer=1))
    interleaved = autoregressive_models.AutoregressiveInterleavedModel(3,[4,4,4],5,transformer,list(range(12)), list(range(12, 12+3)), 12+3)
    pgmz0 = markov_networks.TreeMRF(nvars = 3, nvals = 4, initializer = constants.TensorInitializer.CONSTANT).to_probability_table()
    model = bayesian_networks.BayesianNetwork.join(pgmz0.to_bayesian_network(), interleaved.to_bayesian_network(), shared=dict(enumerate(range(3))))
    sampler = ancestral.BatchAncestralSamplerWithPotentialTables()
    x = sampler.sample(10, model)
    print(x)
