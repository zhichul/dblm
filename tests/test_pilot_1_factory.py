
import unittest

import torch
from dblm.core.modeling import constants, factory

from dblm.utils import seeding


class TestPilot1Factory(unittest.TestCase):

    def setUp(self) -> None:
        seeding.seed(42)
        z0_num_variables = 3
        z0_num_values = 3
        sequence_length = 3
        noise_weight = 0.2
        factor_graph_model, nested_model, bayes_net_model, name = factory.tree_mrf_with_term_frequency_based_transition_and_separate_noise_per_token(  # noqa: F821
        z0_num_variables,
        z0_num_values,
        constants.TensorInitializer.UNIFORM,
        (1-noise_weight, noise_weight),
        sequence_length,
        constants.TensorInitializer.UNIFORM,
        (1-noise_weight, noise_weight))
        self.factor_graph_model = factor_graph_model
        self.nested_model = nested_model
        self.bayes_net_model = bayes_net_model
        self.name = name

    def test_agreement(self):
        ea = self.factor_graph_model.energy((0,1,2,
                                                               2,2,2,1,1,1,2,2,2,
                                                               1,1,1,1,1,1,1,1,1,
                                                               0,0,0,0,0,0,0,1,2,
                                                               0,1,0,0,2,
                                                               1,0,0,1,4,
                                                               1,2,1,2,8,
                                                               ))
        eb = self.factor_graph_model.energy((0,1,2,
                                                               2,2,2,0,1,0,0,2,2,
                                                               1,1,1,1,0,1,1,1,1,
                                                               0,0,0,0,1,0,0,0,2,
                                                               0,1,0,0,0,
                                                               1,0,0,1,4,
                                                               1,2,1,2,8,
                                                               ))
        la = self.bayes_net_model.log_probability((0,1,2,
                                                               2,2,2,1,1,1,2,2,2,
                                                               1,1,1,1,1,1,1,1,1,
                                                               0,0,0,0,0,0,0,1,2,
                                                               0,1,0,0,2,
                                                               1,0,0,1,4,
                                                               1,2,1,2,8,
                                                               ))
        lb = self.bayes_net_model.log_probability((0,1,2,
                                                               2,2,2,0,1,0,0,2,2,
                                                               1,1,1,1,0,1,1,1,1,
                                                               0,0,0,0,1,0,0,0,2,
                                                               0,1,0,0,0,
                                                               1,0,0,1,4,
                                                               1,2,1,2,8,
                                                               ))
        torch.testing.assert_close((ea - eb), (la-lb))

if __name__ == "__main__":
    unittest.main()


