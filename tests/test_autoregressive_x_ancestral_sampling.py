
from dblm.core.modeling import bayesian_networks
from dblm.core.modeling.interleaved_models import autoregressive_x_models
from dblm.utils import seeding


if __name__ == "__main__":
    from dblm.core.modeling import gpt2
    from dblm.core.modeling import markov_networks
    from dblm.core.modeling import constants
    from dblm.core.samplers import ancestral
    seeding.seed(42)
    transformer = gpt2.GPT2LMHeadModel(gpt2.GPT2Config(vocab_size=24 + 1, n_positions=3 + 1 + 5, bos_token_id=12, n_layer=1))
    interleaved = autoregressive_x_models.AutoregressiveXModel(3,[4,4,4],5,transformer,list(range(13,25)), 12)
    pgmz0 = markov_networks.TreeMRF(nvars = 3, nvals = 4, initializer = constants.TensorInitializer.CONSTANT).to_probability_table()
    model = bayesian_networks.BayesianNetwork.join(pgmz0.to_bayesian_network(), interleaved.to_bayesian_network(), shared=dict(enumerate(range(3))))
    sampler = ancestral.BatchAncestralSamplerWithPotentialTables()
    model.eval()
    x = sampler.sample(10, model)
    print(x)
