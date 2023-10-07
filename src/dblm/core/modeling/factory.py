
from __future__ import annotations
from dblm.core.modeling import bayesian_networks, constants, factor_graphs, markov_networks, noise
from dblm.core.modeling.interleaved_models import log_linear_transitions

def tree_mrf_with_term_frequency_based_transition_and_separate_noise_per_token(z0_nvars: int,
                                                                               z0_nvals:int | list[int],
                                                                               z0_param_initializer: constants.TensorInitializer,
                                                                               z0_noise_ratio: tuple[float,float], # fist is signal second is noise
                                                                               zt_sequence_length,
                                                                               zt_param_initializer: constants.TensorInitializer,
                                                                               zt_noise_ratio: tuple[float,float]): # fist is signal second is noise
    if isinstance(z0_nvals, int):
        z0_nvals = [z0_nvals] * z0_nvars
    assert isinstance(z0_nvals, list)
    pgmz0 =  markov_networks.TreeMRF(z0_nvars, z0_nvals, z0_param_initializer)
    pgmz0_noises = noise.NoisyMixture(z0_nvars, z0_nvals, constants.DiscreteNoise.UNIFORM, mixture_ratio=z0_noise_ratio)
    factor_graph_model = pgmz0.to_factor_graph_model()
    bayes_net_model = pgmz0.to_probability_table().to_bayesian_network()

    for _ in range(zt_sequence_length):
        factor_graph_model = factor_graphs.FactorGraph.join(factor_graph_model, pgmz0_noises.to_factor_graph_model(), dict(enumerate(range(z0_nvars))))
        bayes_net_model = bayesian_networks.BayesianNetwork.join(bayes_net_model, pgmz0_noises,dict(enumerate(range(z0_nvars))))
    pgmztxt = log_linear_transitions.FixedLengthDirectedChainWithLogLinearTermFrequencyTransitionAndDeterministicEmissionWithNoiseInZOneCopyForEveryToken(
        z0_nvars,
        z0_nvals,
        zt_sequence_length,
        zt_param_initializer,
        noise=constants.DiscreteNoise.UNIFORM,
        mixture_ratio=zt_noise_ratio,
        separate_noise_distribution_per_state=False)
    bindings = [z0_nvars + i * z0_nvars * 3 + z0_nvars * 2 + j for i in range(zt_sequence_length) for j in range(z0_nvars)]
    # z0 z0 z0 z0 _ _ _ _, _ _ _ _, z0'1 z0'1 z0'1 z0'1 _ _ _ _, _ _ _ _, z0'2 z0'2 z0'2 z0'2 ...
    factor_graph_model = factor_graphs.FactorGraph.join(factor_graph_model, pgmztxt.to_factor_graph_model(), dict(enumerate(bindings)))
    bayes_net_model = bayesian_networks.BayesianNetwork.join(bayes_net_model, pgmztxt, dict(enumerate(bindings)))
    factor_offset_zt = z0_nvars - 1 + 3 * z0_nvars * zt_sequence_length
    var_offset_zt = factor_offset_zt + 1
    nested_model = factor_graphs.BPAutoregressiveIncompleteLikelihoodFactorGraph.from_factor_graph(factor_graph_model, [(factor_offset_zt + i * 5 + 4, var_offset_zt + i * 5 + 4)for i in range(zt_sequence_length)]) # offset_zt is the initial index of the first factor related to t>0, and +1 is the offset
    names = [f"u({i})" for i in range(z0_nvars)]
    for t in range(zt_sequence_length):
        names += [f"unoise({t},{i})" for i in range(z0_nvars)] + [f"uswitch({t},{i})" for i in range(z0_nvars)] + [f"u'({t},{i})" for i in range(z0_nvars)]
    for t in range(zt_sequence_length):
        names += [f"z({t})", f"znoise({t})", f"zswitch({t})", f"z'({t})", f"x({t})"]
    assert factor_graph_model.nvars == len(names)
    return factor_graph_model, nested_model, bayes_net_model, names


def tree_mrf_with_term_frequency_based_transition_and_no_noise(z0_nvars: int,
                                                                z0_nvals:int | list[int],
                                                                z0_param_initializer: constants.TensorInitializer,
                                                                zt_sequence_length,
                                                                zt_param_initializer: constants.TensorInitializer):
    if isinstance(z0_nvals, int):
        z0_nvals = [z0_nvals] * z0_nvars
    assert isinstance(z0_nvals, list)
    pgmz0 =  markov_networks.TreeMRF(z0_nvars, z0_nvals, z0_param_initializer)
    factor_graph_model = pgmz0.to_factor_graph_model()
    bayes_net_model = pgmz0.to_probability_table().to_bayesian_network()

    pgmztxt = log_linear_transitions.FixedLengthDirectedChainWithLogLinearTermFrequencyTransitionAndDeterministicEmission(
        z0_nvars,
        z0_nvals,
        zt_sequence_length,
        zt_param_initializer)
    bindings = [j for j in range(z0_nvars)]
    # z0 z0 z0 z0 _ _ _ _, _ _ _ _, z0'1 z0'1 z0'1 z0'1 _ _ _ _, _ _ _ _, z0'2 z0'2 z0'2 z0'2 ...
    factor_graph_model = factor_graphs.FactorGraph.join(factor_graph_model, pgmztxt.to_factor_graph_model(), dict(enumerate(bindings)))
    bayes_net_model = bayesian_networks.BayesianNetwork.join(bayes_net_model, pgmztxt, dict(enumerate(bindings)))
    factor_offset_zt = z0_nvars - 1
    var_offset_zt = factor_offset_zt + 1
    nested_model = factor_graphs.BPAutoregressiveIncompleteLikelihoodFactorGraph.from_factor_graph(factor_graph_model, [(factor_offset_zt + i * 2 + 1, var_offset_zt + i * 2 + 1)for i in range(zt_sequence_length)]) # offset_zt is the initial index of the first factor related to t>0, and +1 is the offset
    names = [f"u({i})" for i in range(z0_nvars)]
    for t in range(zt_sequence_length):
        names += [f"z({t})", f"x({t})"]
    assert factor_graph_model.nvars == len(names)
    return factor_graph_model, nested_model, bayes_net_model, names
