
from dblm.core.modeling.interleaved_models import autoregressive_x_models
from dblm.utils import seeding


if __name__ == "__main__":
    from dblm.core.modeling import gpt2
    from dblm.core.modeling import markov_networks
    from dblm.core.modeling import constants
    seeding.seed(42)
    transformer = gpt2.GPT2LMHeadModel(gpt2.GPT2Config(vocab_size=24 + 1, n_positions=3 + 1 + 5, bos_token_id=12, n_layer=1))
    pgmxt = autoregressive_x_models.AutoregressiveXModel(3,[4,4,4],5,transformer,list(range(13,25)), 12)
    pgmz0 = markov_networks.TreeMRF(nvars = 3, nvals = 4, initializer = constants.GaussianInitializer(0, 1)) # type:ignore
    model = autoregressive_x_models.LatentWorldAutoregressiveXModel(pgmz0.to_factor_graph_model(), pgmxt)
    assignment = list(enumerate([ 1,  3,  2, 12, 21, 16, 23, 21]))
    model.eval()
    model.log_marginal_probability(assignment)
