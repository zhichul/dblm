
import code
import unittest

import torch
from dblm.core.inferencers import belief_propagation
from dblm.core.modeling import constants, factor_graphs, factory

from dblm.utils import seeding


class TestTensorBP(unittest.TestCase):

    def setUp(self) -> None:
        seeding.seed(42)
        z0_nvars = 3
        seq_length = 5
        self.factor_graph5, _, _, _ = factory.tree_mrf_with_term_frequency_based_transition_and_separate_noise_per_token(
            z0_nvars,
            4,
            constants.TensorInitializer.UNIFORM,
            (4,1),
            seq_length,
            constants.TensorInitializer.UNIFORM,
            (4,1),
        )
        self.factor_graph4 = get_sub_model(self.factor_graph5, z0_nvars, seq_length, 4)
        self.factor_graph3 = get_sub_model(self.factor_graph5, z0_nvars, seq_length, 3)
        self.factor_graph2 = get_sub_model(self.factor_graph5, z0_nvars, seq_length, 2)
        self.factor_graph1 = get_sub_model(self.factor_graph5, z0_nvars, seq_length, 1)
        self.bp = belief_propagation.FactorGraphBeliefPropagation()

    def test_bp1(self):
        with torch.no_grad():
            materialized_switch_bp_results = self.bp.inference(self.factor_graph1, {}, [self.factor_graph1.nvars-1], iterations=1, return_messages=True, renormalize=True, materialize_switch=True, true_parallel=True)
            lazy_switch_bp_results = self.bp.inference(self.factor_graph1, {}, [self.factor_graph1.nvars-1], iterations=1, return_messages=True, renormalize=True, materialize_switch=False, true_parallel=True)
            for m1, m2 in zip(materialized_switch_bp_results.messages_to_variables[0], lazy_switch_bp_results.messages_to_variables[0]): #type:ignore
                if m2 is not None:
                    self.assertTrue((m1.probability_table() == m2.probability_table()).all()) # type:ignore
                else:
                    torch.testing.assert_close(m1.probability_table(), torch.zeros_like(m1.probability_table()).softmax(-1)) # type: ignore
            f0, f1 = self.factor_graph1.factor_functions()[:2]
            m00, m02, m11, m12 = materialized_switch_bp_results.messages_to_variables[0][:4]  #type:ignore
            m00 = m00.probability_table() # type:ignore
            m02 = m02.probability_table() # type:ignore
            m11 = m11.probability_table() # type:ignore
            m12 = m12.probability_table() # type:ignore
            torch.testing.assert_close(m00, f0.logits.logsumexp(1).softmax(-1)) # type: ignore
            torch.testing.assert_close(m02, f0.logits.logsumexp(0).softmax(-1)) # type: ignore
            torch.testing.assert_close(m11, f1.logits.logsumexp(1).softmax(-1)) # type: ignore
            torch.testing.assert_close(m12, f1.logits.logsumexp(0).softmax(-1)) # type: ignore
            m23, m34, m45, m56, m67, m78 = materialized_switch_bp_results.messages_to_variables[0][4:10]  #type:ignore
            m23 = m23.probability_table() # type:ignore
            m34 = m34.probability_table() # type:ignore
            m45 = m45.probability_table() # type:ignore
            m56 = m56.probability_table() # type:ignore
            m67 = m67.probability_table() # type:ignore
            m78 = m78.probability_table() # type:ignore
            torch.testing.assert_close(m23, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m34, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m45, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m56, torch.tensor([0.8, 0.2])) # type: ignore
            torch.testing.assert_close(m67, torch.tensor([0.8, 0.2])) # type: ignore
            torch.testing.assert_close(m78, torch.tensor([0.8, 0.2])) # type: ignore
            m80, m83, m86, m89 = materialized_switch_bp_results.messages_to_variables[0][10:14]  #type:ignore
            m91, m94, m97, m9_10 = materialized_switch_bp_results.messages_to_variables[0][14:18]  #type:ignore
            m10_2, m10_5, m10_8, m10_11 = materialized_switch_bp_results.messages_to_variables[0][18:22]  #type:ignore
            m80 = m80.probability_table() # type:ignore
            m91 = m91.probability_table() # type:ignore
            m10_2 = m10_2.probability_table() # type:ignore
            m83 = m83.probability_table() # type:ignore
            m94 = m94.probability_table() # type:ignore
            m10_5 = m10_5.probability_table() # type:ignore
            m86 = m86.probability_table() # type:ignore
            m97 = m97.probability_table() # type:ignore
            m10_8 = m10_8.probability_table() # type:ignore
            m89 = m89.probability_table() # type:ignore
            m9_10 = m9_10.probability_table() # type:ignore
            m10_11 = m10_11.probability_table() # type:ignore
            torch.testing.assert_close(m80, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m83, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m86, torch.tensor([0.5, 0.5])) # type: ignore
            torch.testing.assert_close(m89, torch.tensor([0.25] * 4)) # type: ignore this is only true first iter due to no incoming messages from anyone
            torch.testing.assert_close(m91, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m94, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m97, torch.tensor([0.5, 0.5])) # type: ignore
            torch.testing.assert_close(m9_10, torch.tensor([0.25] * 4)) # type: ignore this is only true first iter due to no incoming messages from anyone
            torch.testing.assert_close(m10_2, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m10_5, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m10_8, torch.tensor([0.5, 0.5])) # type: ignore
            torch.testing.assert_close(m10_11, torch.tensor([0.25] * 4)) # type: ignore this is only true first iter due to no incoming messages from anyone
            m11_12 = materialized_switch_bp_results.messages_to_variables[0][22]  #type:ignore
            f11 = self.factor_graph1.factor_functions()[11]
            torch.testing.assert_close(m11_12.probability_table(), f11.probability_table()) # type: ignore
            m12_13 = materialized_switch_bp_results.messages_to_variables[0][23]  #type:ignore
            torch.testing.assert_close(m12_13.probability_table(), torch.tensor([1/3] * 3)) # type: ignore
            m13_14 = materialized_switch_bp_results.messages_to_variables[0][24]  #type:ignore
            torch.testing.assert_close(m13_14.probability_table(), torch.tensor([0.8, 0.2])) # type: ignore
            m14_12, m14_13, m14_14, m14_15 = materialized_switch_bp_results.messages_to_variables[0][25:29]  #type:ignore
            torch.testing.assert_close(m14_12.probability_table(), torch.tensor([1/3] * 3)) # type: ignore
            torch.testing.assert_close(m14_13.probability_table(), torch.tensor([1/3] * 3)) # type: ignore
            torch.testing.assert_close(m14_14.probability_table(), torch.tensor([1/2] * 2)) # type: ignore
            torch.testing.assert_close(m14_15.probability_table(), torch.tensor([1/3] * 3)) # type: ignore this is only true first iter due to no incoming messages from anyone
            m15_9, m15_10, m15_11, m15_15, m15_16 = materialized_switch_bp_results.messages_to_variables[0][29:34] # type:ignore
            torch.testing.assert_close(m15_9.probability_table(), torch.tensor([1/4] * 4)) # type: ignore this is only true first iter due to no incoming messages from anyone
            torch.testing.assert_close(m15_10.probability_table(), torch.tensor([1/4] * 4)) # type: ignore this is only true first iter due to no incoming messages from anyone
            torch.testing.assert_close(m15_11.probability_table(), torch.tensor([1/4] * 4)) # type: ignore this is only true first iter due to no incoming messages from anyone
            torch.testing.assert_close(m15_15.probability_table(), torch.tensor([1/3] * 3)) # type: ignore this is only true first iter due to no incoming messages from anyone
            torch.testing.assert_close(m15_16.probability_table(), torch.tensor([1/12] * 12)) # type: ignore this is only true first iter due to no incoming messages from anyone
            self.assertEqual(len(materialized_switch_bp_results.messages_to_variables[0]), 34) # type:ignore

            materialized_switch_bp_results = self.bp.inference(self.factor_graph1, {}, [self.factor_graph1.nvars-1], iterations=1, return_messages=True, renormalize=True, materialize_switch=True, true_parallel=True,
                        messages_to_variables=materialized_switch_bp_results.messages_to_variables[0], messages_to_factors=materialized_switch_bp_results.messages_to_factors[0]) # type:ignore
            lazy_switch_bp_results = self.bp.inference(self.factor_graph1, {}, [self.factor_graph1.nvars-1], iterations=1, return_messages=True, renormalize=True, materialize_switch=False, true_parallel=True,
                        messages_to_variables=lazy_switch_bp_results.messages_to_variables[0], messages_to_factors=lazy_switch_bp_results.messages_to_factors[0]) # type:ignore
            for m1, m2 in zip(materialized_switch_bp_results.messages_to_variables[0], lazy_switch_bp_results.messages_to_variables[0]): #type:ignore
                if m2 is not None:
                    self.assertTrue((m1.probability_table() == m2.probability_table()).all()) # type:ignore
                else:
                    torch.testing.assert_close(m1.probability_table(), torch.zeros_like(m1.probability_table()).softmax(-1)) # type: ignore
            f0, f1 = self.factor_graph1.factor_functions()[:2]
            m00, m02, m11, m12 = materialized_switch_bp_results.messages_to_factors[0][:4]  #type:ignore
            m00 = m00.renormalize().probability_table() # type:ignore
            m02 = m02.renormalize().probability_table() # type:ignore
            m11 = m11.renormalize().probability_table() # type:ignore
            m12 = m12.renormalize().probability_table() # type:ignore
            torch.testing.assert_close(m00, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m02, f1.logits.logsumexp(0).softmax(-1)) # type: ignore
            torch.testing.assert_close(m11, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m12, f0.logits.logsumexp(0).softmax(-1)) # type: ignore
            m23, m34, m45, m56, m67, m78 = materialized_switch_bp_results.messages_to_factors[0][4:10]  #type:ignore
            m23 = m23.renormalize().probability_table() # type:ignore
            m34 = m34.renormalize().probability_table() # type:ignore
            m45 = m45.renormalize().probability_table() # type:ignore
            m56 = m56.renormalize().probability_table() # type:ignore
            m67 = m67.renormalize().probability_table() # type:ignore
            m78 = m78.renormalize().probability_table() # type:ignore
            torch.testing.assert_close(m23, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m34, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m45, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m56, torch.tensor([0.5, 0.5])) # type: ignore
            torch.testing.assert_close(m67, torch.tensor([0.5, 0.5])) # type: ignore
            torch.testing.assert_close(m78, torch.tensor([0.5, 0.5])) # type: ignore
            m80, m83, m86, m89 = materialized_switch_bp_results.messages_to_factors[0][10:14]  #type:ignore
            m91, m94, m97, m9_10 = materialized_switch_bp_results.messages_to_factors[0][14:18]  #type:ignore
            m10_2, m10_5, m10_8, m10_11 = materialized_switch_bp_results.messages_to_factors[0][18:22]  #type:ignore
            m80 = m80.renormalize().probability_table() # type:ignore
            m91 = m91.renormalize().probability_table() # type:ignore
            m10_2 = m10_2.renormalize().probability_table() # type:ignore
            m83 = m83.renormalize().probability_table() # type:ignore
            m94 = m94.renormalize().probability_table() # type:ignore
            m10_5 = m10_5.renormalize().probability_table() # type:ignore
            m86 = m86.renormalize().probability_table() # type:ignore
            m97 = m97.renormalize().probability_table() # type:ignore
            m10_8 = m10_8.renormalize().probability_table() # type:ignore
            m89 = m89.renormalize().probability_table() # type:ignore
            m9_10 = m9_10.renormalize().probability_table() # type:ignore
            m10_11 = m10_11.renormalize().probability_table() # type:ignore
            torch.testing.assert_close(m80, f0.logits.logsumexp(1).softmax(-1)) # type: ignore
            torch.testing.assert_close(m83, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m86, torch.tensor([0.8, 0.2])) # type: ignore
            torch.testing.assert_close(m89, torch.tensor([0.25] * 4)) # type: ignore this is only true first iter due to no incoming messages from anyone
            torch.testing.assert_close(m91, f1.logits.logsumexp(1).softmax(-1)) # type: ignore
            torch.testing.assert_close(m94, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m97, torch.tensor([0.8, 0.2])) # type: ignore
            torch.testing.assert_close(m9_10, torch.tensor([0.25] * 4)) # type: ignore this is only true first iter due to no incoming messages from anyone
            torch.testing.assert_close(m10_2, (f0.logits.logsumexp(0) + f1.logits.logsumexp(0)).softmax(-1)) # type: ignore
            torch.testing.assert_close(m10_5, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m10_8, torch.tensor([0.8, 0.2])) # type: ignore
            torch.testing.assert_close(m10_11, torch.tensor([0.25] * 4)) # type: ignore this is only true first iter due to no incoming messages from anyone
            m11_12 = materialized_switch_bp_results.messages_to_factors[0][22]  #type:ignore
            f11 = self.factor_graph1.factor_functions()[11]
            torch.testing.assert_close(m11_12.probability_table(), torch.tensor([1/3] *3)) # type: ignore
            m12_13 = materialized_switch_bp_results.messages_to_factors[0][23]  #type:ignore
            torch.testing.assert_close(m12_13.probability_table(), torch.tensor([1/3] * 3)) # type: ignore
            m13_14 = materialized_switch_bp_results.messages_to_factors[0][24]  #type:ignore
            torch.testing.assert_close(m13_14.probability_table(), torch.tensor([0.5, 0.5])) # type: ignore
            m14_12, m14_13, m14_14, m14_15 = materialized_switch_bp_results.messages_to_factors[0][25:29]  #type:ignore
            torch.testing.assert_close(m14_12.probability_table(), f11.probability_table()) # type: ignore
            torch.testing.assert_close(m14_13.probability_table(), torch.tensor([1/3] * 3)) # type: ignore
            torch.testing.assert_close(m14_14.probability_table(), torch.tensor([0.8, 0.2])) # type: ignore
            torch.testing.assert_close(m14_15.probability_table(), torch.tensor([1/3] * 3)) # type: ignore this is only true first iter due to no incoming messages from anyone

            m15_9, m15_10, m15_11, m15_15, m15_16 = materialized_switch_bp_results.messages_to_factors[0][29:34] # type:ignore
            torch.testing.assert_close(m15_9.probability_table(), torch.tensor([1/4] * 4)) # type: ignore this is only true first iter due to no incoming messages from anyone
            torch.testing.assert_close(m15_10.probability_table(), torch.tensor([1/4] * 4)) # type: ignore this is only true first iter due to no incoming messages from anyone
            torch.testing.assert_close(m15_11.probability_table(), torch.tensor([1/4] * 4)) # type: ignore this is only true first iter due to no incoming messages from anyone
            torch.testing.assert_close(m15_15.probability_table(), torch.tensor([1/3] * 3)) # type: ignore this is only true first iter due to no incoming messages from anyone
            self.assertEqual(m15_16, None) # type: ignore this is only true first iter due to no incoming messages from anyone
            self.assertEqual(len(materialized_switch_bp_results.messages_to_factors[0]), 34) # type:ignore

            materialized_switch_bp_results = self.bp.inference(self.factor_graph1, {}, [self.factor_graph1.nvars-1], iterations=1, return_messages=True, renormalize=True, materialize_switch=True, true_parallel=True,
                        messages_to_variables=materialized_switch_bp_results.messages_to_variables[0], messages_to_factors=materialized_switch_bp_results.messages_to_factors[0]) # type:ignore
            lazy_switch_bp_results = self.bp.inference(self.factor_graph1, {}, [self.factor_graph1.nvars-1], iterations=1, return_messages=True, renormalize=True, materialize_switch=False, true_parallel=True,
                        messages_to_variables=lazy_switch_bp_results.messages_to_variables[0], messages_to_factors=lazy_switch_bp_results.messages_to_factors[0]) # type:ignore
            for m1, m2 in zip(materialized_switch_bp_results.messages_to_variables[0], lazy_switch_bp_results.messages_to_variables[0]): #type:ignore
                if m2 is not None:
                    self.assertTrue((m1.probability_table() == m2.probability_table()).all()) # type:ignore
                else:
                    torch.testing.assert_close(m1.probability_table(), torch.zeros_like(m1.probability_table()).softmax(-1)) # type: ignore
            f0, f1 = self.factor_graph1.factor_functions()[:2]
            m00, m02, m11, m12 = materialized_switch_bp_results.messages_to_variables[0][:4]  #type:ignore
            m00 = m00.probability_table() # type:ignore
            m02 = m02.probability_table() # type:ignore
            m11 = m11.probability_table() # type:ignore
            m12 = m12.probability_table() # type:ignore
            torch.testing.assert_close(m00, (f0.logits[:,None,:] + f1.logits[None,:,:]).logsumexp(-1).logsumexp(-1).softmax(-1)) # type: ignore
            torch.testing.assert_close(m02, f0.logits.logsumexp(0).softmax(-1)) # type: ignore
            torch.testing.assert_close(m11, (f0.logits[:,None,:] + f1.logits[None,:,:]).logsumexp(-1).logsumexp(0).softmax(-1)) # type: ignore
            torch.testing.assert_close(m12, f1.logits.logsumexp(0).softmax(-1)) # type: ignore
            m23, m34, m45, m56, m67, m78 = materialized_switch_bp_results.messages_to_variables[0][4:10]  #type:ignore
            m23 = m23.probability_table() # type:ignore
            m34 = m34.probability_table() # type:ignore
            m45 = m45.probability_table() # type:ignore
            m56 = m56.probability_table() # type:ignore
            m67 = m67.probability_table() # type:ignore
            m78 = m78.probability_table() # type:ignore
            torch.testing.assert_close(m23, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m34, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m45, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m56, torch.tensor([0.8, 0.2])) # type: ignore
            torch.testing.assert_close(m67, torch.tensor([0.8, 0.2])) # type: ignore
            torch.testing.assert_close(m78, torch.tensor([0.8, 0.2])) # type: ignore
            m80, m83, m86, m89 = materialized_switch_bp_results.messages_to_variables[0][10:14]  #type:ignore
            m91, m94, m97, m9_10 = materialized_switch_bp_results.messages_to_variables[0][14:18]  #type:ignore
            m10_2, m10_5, m10_8, m10_11 = materialized_switch_bp_results.messages_to_variables[0][18:22]  #type:ignore
            m80 = m80.probability_table() # type:ignore
            m91 = m91.probability_table() # type:ignore
            m10_2 = m10_2.probability_table() # type:ignore
            m83 = m83.probability_table() # type:ignore
            m94 = m94.probability_table() # type:ignore
            m10_5 = m10_5.probability_table() # type:ignore
            m86 = m86.probability_table() # type:ignore
            m97 = m97.probability_table() # type:ignore
            m10_8 = m10_8.probability_table() # type:ignore
            m89 = m89.probability_table() # type:ignore
            m9_10 = m9_10.probability_table() # type:ignore
            m10_11 = m10_11.probability_table() # type:ignore
            torch.testing.assert_close(m80, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m83, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m86, torch.tensor([0.5, 0.5])) # type: ignore
            torch.testing.assert_close(m89, f0.logits.logsumexp(1).softmax(-1) * 0.8 + torch.tensor([0.25] * 4) * 0.2) # type: ignore this is only true first iter due to no incoming messages from anyone
            torch.testing.assert_close(m91, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m94, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m97, torch.tensor([0.5, 0.5])) # type: ignore
            torch.testing.assert_close(m9_10, f1.logits.logsumexp(1).softmax(-1) * 0.8 + torch.tensor([0.25] * 4) * 0.2) # type: ignore this is only true first iter due to no incoming messages from anyone
            torch.testing.assert_close(m10_2, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m10_5, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m10_8, torch.tensor([0.5, 0.5])) # type: ignore
            torch.testing.assert_close(m10_11, (f0.logits.logsumexp(0) + f1.logits.logsumexp(0)).softmax(-1) * 0.8 + torch.tensor([0.25] * 4) * 0.2) # type: ignore this is only true first iter due to no incoming messages from anyone
            m11_12 = materialized_switch_bp_results.messages_to_variables[0][22]  #type:ignore
            f11 = self.factor_graph1.factor_functions()[11]
            torch.testing.assert_close(m11_12.probability_table(), f11.probability_table()) # type: ignore
            m12_13 = materialized_switch_bp_results.messages_to_variables[0][23]  #type:ignore
            torch.testing.assert_close(m12_13.probability_table(), torch.tensor([1/3] * 3)) # type: ignore
            m13_14 = materialized_switch_bp_results.messages_to_variables[0][24]  #type:ignore
            torch.testing.assert_close(m13_14.probability_table(), torch.tensor([0.8, 0.2])) # type: ignore
            m14_12, m14_13, m14_14, m14_15 = materialized_switch_bp_results.messages_to_variables[0][25:29]  #type:ignore
            torch.testing.assert_close(m14_12.probability_table(), torch.tensor([1/3] * 3)) # type: ignore
            torch.testing.assert_close(m14_13.probability_table(), torch.tensor([1/3] * 3)) # type: ignore
            torch.testing.assert_close(m14_14.probability_table(), torch.tensor([1/2] * 2)) # type: ignore
            torch.testing.assert_close(m14_15.probability_table(), f11.probability_table() * 0.8 + torch.tensor([1/3] * 3) * 0.2) # type: ignore this is only true first iter due to no incoming messages from anyone
            m15_9, m15_10, m15_11, m15_15, m15_16 = materialized_switch_bp_results.messages_to_variables[0][29:34] # type:ignore
            torch.testing.assert_close(m15_9.probability_table(), torch.tensor([1/4] * 4)) # type: ignore this is only true first iter due to no incoming messages from anyone
            torch.testing.assert_close(m15_10.probability_table(), torch.tensor([1/4] * 4)) # type: ignore this is only true first iter due to no incoming messages from anyone
            torch.testing.assert_close(m15_11.probability_table(), torch.tensor([1/4] * 4)) # type: ignore this is only true first iter due to no incoming messages from anyone
            torch.testing.assert_close(m15_15.probability_table(), torch.tensor([1/3] * 3)) # type: ignore this is only true first iter due to no incoming messages from anyone
            torch.testing.assert_close(m15_16.probability_table(), torch.tensor([1/12] * 12)) # type: ignore this is only true first iter due to no incoming messages from anyone
            self.assertEqual(len(materialized_switch_bp_results.messages_to_variables[0]), 34) # type:ignore

            # msg to z0's noise should be good now
            materialized_switch_bp_results = self.bp.inference(self.factor_graph1, {}, [self.factor_graph1.nvars-1], iterations=2, return_messages=True, renormalize=True, materialize_switch=True, true_parallel=True,
                        messages_to_variables=materialized_switch_bp_results.messages_to_variables[-1], messages_to_factors=materialized_switch_bp_results.messages_to_factors[-1]) # type:ignore
            lazy_switch_bp_results = self.bp.inference(self.factor_graph1, {}, [self.factor_graph1.nvars-1], iterations=2, return_messages=True, renormalize=True, materialize_switch=False, true_parallel=True,
                        messages_to_variables=lazy_switch_bp_results.messages_to_variables[-1], messages_to_factors=lazy_switch_bp_results.messages_to_factors[-1]) # type:ignore
            for m1, m2 in zip(materialized_switch_bp_results.messages_to_variables[0], lazy_switch_bp_results.messages_to_variables[0]): #type:ignore
                if m2 is not None:
                    self.assertTrue((m1.probability_table() == m2.probability_table()).all()) # type:ignore
                else:
                    torch.testing.assert_close(m1.probability_table(), torch.zeros_like(m1.probability_table()).softmax(-1)) # type: ignore
            m80, m83, m86, m89 = materialized_switch_bp_results.messages_to_variables[1][10:14]  #type:ignore
            m91, m94, m97, m9_10 = materialized_switch_bp_results.messages_to_variables[1][14:18]  #type:ignore
            m10_2, m10_5, m10_8, m10_11 = materialized_switch_bp_results.messages_to_variables[1][18:22]  #type:ignore
            m80 = m80.probability_table() # type:ignore
            m91 = m91.probability_table() # type:ignore
            m10_2 = m10_2.probability_table() # type:ignore
            m83 = m83.probability_table() # type:ignore
            m94 = m94.probability_table() # type:ignore
            m10_5 = m10_5.probability_table() # type:ignore
            m86 = m86.probability_table() # type:ignore
            m97 = m97.probability_table() # type:ignore
            m10_8 = m10_8.probability_table() # type:ignore
            m89 = m89.probability_table() # type:ignore
            m9_10 = m9_10.probability_table() # type:ignore
            m10_11 = m10_11.probability_table() # type:ignore
            torch.testing.assert_close(m80, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m83, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m86, torch.tensor([0.5, 0.5])) # type: ignore
            torch.testing.assert_close(m89, (f0.logits[:,None,:] + f1.logits[None,:,:]).logsumexp(-1).logsumexp(-1).softmax(-1) * 0.8 + torch.tensor([0.25] * 4) * 0.2) # type: ignore this is only true first iter due to no incoming messages from anyone
            torch.testing.assert_close(m91, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m94, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m97, torch.tensor([0.5, 0.5])) # type: ignore
            torch.testing.assert_close(m9_10, (f0.logits[:,None,:] + f1.logits[None,:,:]).logsumexp(-1).logsumexp(0).softmax(-1) * 0.8 + torch.tensor([0.25] * 4) * 0.2) # type: ignore this is only true first iter due to no incoming messages from anyone
            torch.testing.assert_close(m10_2, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m10_5, torch.tensor([0.25] * 4)) # type: ignore
            torch.testing.assert_close(m10_8, torch.tensor([0.5, 0.5])) # type: ignore
            torch.testing.assert_close(m10_11, (f0.logits[:,None,:] + f1.logits[None,:,:]).logsumexp(0).logsumexp(0).softmax(-1) * 0.8 + torch.tensor([0.25] * 4) * 0.2) # type: ignore this is only true first iter due to no incoming messages from anyone

            materialized_switch_bp_results = self.bp.inference(self.factor_graph1, {}, [self.factor_graph1.nvars-1], iterations=2, return_messages=True, renormalize=True, materialize_switch=True, true_parallel=True,
                        messages_to_variables=materialized_switch_bp_results.messages_to_variables[-1], messages_to_factors=materialized_switch_bp_results.messages_to_factors[-1]) # type:ignore
            lazy_switch_bp_results = self.bp.inference(self.factor_graph1, {}, [self.factor_graph1.nvars-1], iterations=2, return_messages=True, renormalize=True, materialize_switch=False, true_parallel=True,
                        messages_to_variables=lazy_switch_bp_results.messages_to_variables[-1], messages_to_factors=lazy_switch_bp_results.messages_to_factors[-1]) # type:ignore
            for m1, m2 in zip(materialized_switch_bp_results.messages_to_variables[0], lazy_switch_bp_results.messages_to_variables[0]): #type:ignore
                if m2 is not None:
                    torch.testing.assert_close(m1.probability_table(), m2.probability_table()) # type:ignore
                else:
                    torch.testing.assert_close(m1.probability_table(), torch.zeros_like(m1.probability_table()).softmax(-1)) # type: ignore

            m80, m83, m86, m89 = materialized_switch_bp_results.messages_to_variables[1][10:14]  #type:ignore
            m91, m94, m97, m9_10 = materialized_switch_bp_results.messages_to_variables[1][14:18]  #type:ignore
            m10_2, m10_5, m10_8, m10_11 = materialized_switch_bp_results.messages_to_variables[1][18:22]  #type:ignore
            m15_9, m15_10, m15_11, m15_15, m15_16 = materialized_switch_bp_results.messages_to_variables[1][29:34] # type:ignore
            torch.testing.assert_close(m15_9.probability_table(), torch.tensor([1/4] * 4)) # type: ignore this is only true first iter due to no incoming messages from anyone
            torch.testing.assert_close(m15_10.probability_table(), torch.tensor([1/4] * 4)) # type: ignore this is only true first iter due to no incoming messages from anyone
            torch.testing.assert_close(m15_11.probability_table(), torch.tensor([1/4] * 4)) # type: ignore this is only true first iter due to no incoming messages from anyone
            torch.testing.assert_close(m15_15.probability_table(), torch.tensor([1/3] * 3)) # type: ignore this is only true first iter due to no incoming messages from anyone
            z1_dist = f11.probability_table() * 0.8 + torch.tensor([1/3] * 3) * 0.2 # type:ignore
            emission = (z1_dist[:,None] * torch.stack([m89.probability_table(), m9_10.probability_table(), m10_11.probability_table()], dim=0)).reshape(-1) # type:ignore
            torch.testing.assert_close(m15_16.probability_table(), emission) # type: ignore this is only true first iter due to no incoming messages from anyone
            self.assertEqual(len(materialized_switch_bp_results.messages_to_variables[0]), 34) # type:ignore
            materialized_switch_bp_results = self.bp.inference(self.factor_graph1, {}, [self.factor_graph1.nvars-1], iterations=100, return_messages=True, renormalize=True, materialize_switch=True, true_parallel=True,
                        messages_to_variables=materialized_switch_bp_results.messages_to_variables[-1], messages_to_factors=materialized_switch_bp_results.messages_to_factors[-1]) # type:ignore
            lazy_switch_bp_results = self.bp.inference(self.factor_graph1, {}, [self.factor_graph1.nvars-1], iterations=100, return_messages=True, renormalize=True, materialize_switch=False, true_parallel=True,
                        messages_to_variables=lazy_switch_bp_results.messages_to_variables[-1], messages_to_factors=lazy_switch_bp_results.messages_to_factors[-1]) # type:ignore
            for m1, m2 in zip(materialized_switch_bp_results.messages_to_variables[0], lazy_switch_bp_results.messages_to_variables[0]): #type:ignore
                if m2 is not None:
                    torch.testing.assert_close(m1.probability_table(), m2.probability_table()) # type:ignore
                else:
                    torch.testing.assert_close(m1.probability_table(), torch.zeros_like(m1.probability_table()).softmax(-1)) # type: ignore
            torch.testing.assert_close(m15_16.probability_table(), emission) # type: ignore this is only true first iter due to no incoming messages from anyone
            msg_to_vars = None
            msg_to_factors = None
            for i in range(10):
                lazy_switch_bp_results = self.bp.inference(self.factor_graph1, {}, [self.factor_graph1.nvars-1], iterations=1, return_messages=True, renormalize=True, materialize_switch=False, true_parallel=True,
                            messages_to_variables=msg_to_vars, messages_to_factors=msg_to_factors) # type:ignore
                msg_to_vars = lazy_switch_bp_results.messages_to_variables[0] # type:ignore
                msg_to_factors = lazy_switch_bp_results.messages_to_factors[0] # type:ignore
                print(f"after {i+1} iters", lazy_switch_bp_results.query_marginals[0].probability_table())

    def test_bp2(self):
        msg_to_vars = None
        msg_to_factors = None
        last_table = None
        for i in range(20):
            lazy_switch_bp_results = self.bp.inference(self.factor_graph2, {25: torch.tensor([0, 5, 10])}, [self.factor_graph2.nvars-1], iterations=1, return_messages=True, renormalize=True, materialize_switch=False, true_parallel=True,
                        messages_to_variables=msg_to_vars, messages_to_factors=msg_to_factors) # type:ignore
            msg_to_vars = lazy_switch_bp_results.messages_to_variables[0] # type:ignore
            msg_to_factors = lazy_switch_bp_results.messages_to_factors[0] # type:ignore
            if last_table is not None:
                print(f"after {i+1} iters (delta = {(last_table - lazy_switch_bp_results.query_marginals[0].probability_table()).abs().sum(-1)})")
                # print(lazy_switch_bp_results.query_marginals[0].probability_table())

            last_table = lazy_switch_bp_results.query_marginals[0].probability_table()
        m25_26 = lazy_switch_bp_results.messages_to_variables[0][54].probability_table() # type:ignore
        m17_18 = lazy_switch_bp_results.messages_to_variables[0][31].probability_table() # type:ignore
        m18_19 = lazy_switch_bp_results.messages_to_variables[0][35].probability_table() # type:ignore
        m19_20 = lazy_switch_bp_results.messages_to_variables[0][39].probability_table() # type:ignore
        noisy_switch_marginal = m25_26 * 0.8 + torch.tensor([[1/3] * 3]) * 0.2
        noisy_z0_marginal = torch.stack([m17_18, m18_19, m19_20], dim=1)
        emission = (noisy_switch_marginal[:,:,None] * noisy_z0_marginal).reshape(noisy_z0_marginal.size(0), -1)
        computed = lazy_switch_bp_results.messages_to_variables[0][-1].probability_table() # type:ignore
        torch.testing.assert_close(emission, computed)


def get_sub_model(model, z0_nvars, seq_length, sub_seq_length):
    factor_functions = model.factor_functions()
    factor_variables = model.factor_variables()
    subset_factor_functions = factor_functions[:z0_nvars-1] + factor_functions[z0_nvars-1:z0_nvars-1+z0_nvars*3*sub_seq_length] + factor_functions[z0_nvars-1+z0_nvars*3*seq_length:z0_nvars-1+z0_nvars*3*seq_length+5*sub_seq_length]
    subset_factor_variables = factor_variables[:z0_nvars-1] + factor_variables[z0_nvars-1:z0_nvars-1+z0_nvars*3*sub_seq_length] + factor_variables[z0_nvars-1+z0_nvars*3*seq_length:z0_nvars-1+z0_nvars*3*seq_length+5*sub_seq_length]
    subset_variables = sorted(set(sum(subset_factor_variables, tuple())))
    subset_factor_variables = [tuple(subset_variables.index(var) for var in factor_vars) for factor_vars in subset_factor_variables]
    submodel_nvars = len(subset_variables)
    submodel = factor_graphs.FactorGraph(submodel_nvars, [model.nvals[var] for var in subset_variables], subset_factor_variables, subset_factor_functions)
    return submodel

if __name__ == "__main__":
    # test = TestTensorBP()
    # test.setUp()
    # test.test_bp2()

    unittest.main()
