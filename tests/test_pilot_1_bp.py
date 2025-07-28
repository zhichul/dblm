
import code
import os
import unittest

import torch
from dblm.core.inferencers import belief_propagation
from dblm.core.modeling import constants, factor_graphs, factory

from dblm.experiments import data


class TestPilot1BP(unittest.TestCase):

    def setUp(self) -> None:
        DATA_ROOT = "./experiments/pilot_study_1/data/evaluating_approximation"
        self.factor_graph, self.nested, self.bayes_net, self.names = factory.tree_mrf_with_term_frequency_based_transition_and_separate_noise_per_token(
            2,
            2, constants.TensorInitializer.UNIFORM,
            (4,0.0),
            3,
            constants.TensorInitializer.UNIFORM,
            (4,0.0)
        ) # data.load_model(os.path.join(DATA_ROOT, f"{11}", "ground_truth_models"))
        self.other_factor_graph, self.other_nested, self.other_bayes_net, self.names = factory.tree_mrf_with_term_frequency_based_transition_and_separate_noise_per_token(
            2,
            2, constants.TensorInitializer.UNIFORM,
            (4,0.0),
            3,
            constants.TensorInitializer.UNIFORM,
            (4,0.0)
        )
        for p in self.factor_graph.parameters():
            p.requires_grad = False
        for p in self.other_factor_graph.parameters():
            p.requires_grad = False


    def get_sub_model(self, model):
        factor_functions = model.factor_functions()
        factor_variables = model.factor_variables()
        # subset_factor_functions = factor_functions[:7] + factor_functions[7:31] + factor_functions[247:252]
        # subset_factor_variables = factor_variables[:7] + factor_variables[7:31] + factor_variables[247:252]
        subset_factor_functions = factor_functions[:1] + factor_functions[1:7+6] + factor_functions[19:24+5]
        subset_factor_variables = factor_variables[:1] + factor_variables[1:7+6] + factor_variables[19:24+5]
        subset_variables = sorted(set(sum(subset_factor_variables, tuple())))
        subset_factor_variables = [tuple(subset_variables.index(var) for var in factor_vars) for factor_vars in subset_factor_variables]
        submodel_nvars = len(subset_variables)
        submodel = factor_graphs.FactorGraph(submodel_nvars, [model.nvals[var] for var in subset_variables], subset_factor_variables, subset_factor_functions) # type:ignore
        return submodel

    def test_bp(self):
        bp = belief_propagation.FactorGraphBeliefPropagation()
        submodel = self.get_sub_model(self.factor_graph)
        submodel_materialized = submodel.condition_on({18:3}).to_probability_table()
        dist = submodel_materialized.marginalize_over(range(submodel.nvars-1))
        print(dist.renormalize().probability_table())
        results1 = bp.inference(submodel, {18:torch.tensor([3])}, [submodel.nvars-1], iterations=10, materialize_switch=True).query_marginals[0].probability_table()
        results2 = bp.inference(submodel, {18:torch.tensor([3])}, [submodel.nvars-1], iterations=10, materialize_switch=False).query_marginals[0].probability_table()

        other_submodel = self.get_sub_model(self.other_factor_graph)
        other_submodel_materialized = other_submodel.condition_on({18:3}).to_probability_table()
        other_dist = other_submodel_materialized.marginalize_over(range(other_submodel.nvars-1))
        print(other_dist.renormalize().probability_table())
        other_results1 = bp.inference(other_submodel, {18:torch.tensor([3])}, [other_submodel.nvars-1], iterations=10, materialize_switch=True).query_marginals[0].probability_table()
        other_results2 = bp.inference(other_submodel, {18:torch.tensor([3])}, [other_submodel.nvars-1], iterations=10, materialize_switch=False).query_marginals[0].probability_table()
        e = (results1 * results1.log().clamp_min(-1e10)).sum()
        ce = (results1 * other_results1.log().clamp_min(-1e10)).sum()
        print(e.item(), ce.item())

    def test_convergence_of_bp(self):
        DATA_ROOT = "./experiments/pilot_study_1/data/evaluating_approximation_gaussian_init"
        f, n, b = data.load_model(os.path.join(DATA_ROOT, "11", "ground_truth_models"))
        data_matrix = data.DataMatrix(os.path.join(DATA_ROOT, "11", "samples", "sample.csv"), os.path.join(DATA_ROOT, "11", "ground_truth_models", "mapping.json"))
        indices, assignments = data_matrix.filter_variables_by_name_rule(lambda name: name.startswith("x"))
        sample = assignments[:100]
        bp = belief_propagation.FactorGraphBeliefPropagation()
        msg_to_vars = None
        msg_to_factors = None
        last_table = None
        observation = {ind: sample[:, i] for i, ind in enumerate(indices[:-1])} # type:ignore
        for i in range(200):
            lazy_switch_bp_results = bp.inference(f, observation, [f.nvars-1], iterations=1, return_messages=True, renormalize=True, materialize_switch=False,  # type:ignore
                        messages_to_variables=msg_to_vars, messages_to_factors=msg_to_factors) # type:ignore
            msg_to_vars = lazy_switch_bp_results.messages_to_variables[0] # type:ignore
            msg_to_factors = lazy_switch_bp_results.messages_to_factors[0] # type:ignore
            if last_table is not None:
                delta = (last_table - lazy_switch_bp_results.query_marginals[0].probability_table()).abs().sum(-1)
                if (delta != 0).all():
                    print(f"after {i+1} iters (delta = {delta})")
                # print(lazy_switch_bp_results.query_marginals[0].probability_table())

            last_table = lazy_switch_bp_results.query_marginals[0].probability_table()
if __name__ == "__main__":
    unittest.main()
    # test = TestPilot1BP()
    # test.test_convergence_of_bp()



