from dblm.core.modeling import constants, markov_networks
from dblm.core.samplers.tree import BatchTreeSampler
from dblm.utils import seeding


seeding.seed(42)
tree = markov_networks.TreeMRF(10, 4, constants.GaussianInitializer(0,1,-3,3)) # type:ignore
tree.to("cuda:0")
# for factor in tree._factor_functions:
#     factor._logits.data *= 0 # type:ignore
sampler = BatchTreeSampler()
sample = sampler.sample(100000, tree.to_factor_graph_model())
breakpoint()
