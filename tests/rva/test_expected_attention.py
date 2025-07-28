import unittest
from dblm.rva.discrete import independent

from dblm.utils import seeding
from dblm.rva import expected_attention
import numpy as np
from math import inf

class TestIndependent(unittest.TestCase):

    def setUp(self) -> None:
        self.independent = independent.Independent(
            nvars=3,
            nvals=np.array([2,3,4]),
            dists=[np.array([0.2,0.8]), np.array([0.3,0.1,0.6]), np.array([0.1,0.8,0.07,0.03])]
        )

    def test_independent(self):
        unnormalized_attention_weights = np.array([[1,2,3,4,5,6,7,8,9]])
        log_unnormalized_attention_weights = np.log(unnormalized_attention_weights)
        lp, ass = self.independent.log_probability_table()
        ea = expected_attention.expected_attention(log_unnormalized_attention_weights, lp, ass)
        masked_weights = np.array([
                [1.,0,3,0,0,6,0,0,0],
                [1,0,3,0,0,0,7,0,0],
                [1,0,3,0,0,0,0,8,0],
                [1,0,3,0,0,0,0,0,9],
                [1,0,0,4,0,6,0,0,0],
                [1,0,0,4,0,0,7,0,0],
                [1,0,0,4,0,0,0,8,0],
                [1,0,0,4,0,0,0,0,9],
                [1,0,0,0,5,6,0,0,0],
                [1,0,0,0,5,0,7,0,0],
                [1,0,0,0,5,0,0,8,0],
                [1,0,0,0,5,0,0,0,9],
                [0,2,3,0,0,6,0,0,0],
                [0,2,3,0,0,0,7,0,0],
                [0,2,3,0,0,0,0,8,0],
                [0,2,3,0,0,0,0,0,9],
                [0,2,0,4,0,6,0,0,0],
                [0,2,0,4,0,0,7,0,0],
                [0,2,0,4,0,0,0,8,0],
                [0,2,0,4,0,0,0,0,9],
                [0,2,0,0,5,6,0,0,0],
                [0,2,0,0,5,0,7,0,0],
                [0,2,0,0,5,0,0,8,0],
                [0,2,0,0,5,0,0,0,9],
            ])
        ea_ref = (masked_weights / masked_weights.sum(axis=1, keepdims=True) * np.exp(lp[:,None])).sum(axis=0)[None,:]
        np.testing.assert_allclose(
            ea_ref, ea)
        albo = expected_attention.albo(log_unnormalized_attention_weights, self.independent.log_marginals(), self.independent.log_marginals_conditioned_on_one())
        albo_ref = np.array([[
            1 * 0.2 / (1 + 0.3 * 3 + 0.1 * 4 + 0.6 * 5 + 0.1 * 6 + 0.8 * 7 + 0.07 * 8 + 0.03 * 9),
            2 * 0.8 / (2 + 0.3 * 3 + 0.1 * 4 + 0.6 * 5 + 0.1 * 6 + 0.8 * 7 + 0.07 * 8 + 0.03 * 9),
            3 * 0.3 / (1 * 0.2 + 2 * 0.8 + 3 + 0.1 * 6 + 0.8 * 7 + 0.07 * 8 + 0.03 * 9),
            4 * 0.1 / (1 * 0.2 + 2 * 0.8 + 4 + 0.1 * 6 + 0.8 * 7 + 0.07 * 8 + 0.03 * 9),
            5 * 0.6 / (1 * 0.2 + 2 * 0.8 + 5 + 0.1 * 6 + 0.8 * 7 + 0.07 * 8 + 0.03 * 9),
            6 * 0.1 / (1 * 0.2 + 2 * 0.8 + 0.3 * 3 + 0.1 * 4 + 0.6 * 5 + 6),
            7 * 0.8 / (1 * 0.2 + 2 * 0.8 + 0.3 * 3 + 0.1 * 4 + 0.6 * 5 + 7),
            8 * 0.07 / (1 * 0.2 + 2 * 0.8 + 0.3 * 3 + 0.1 * 4 + 0.6 * 5 + 8),
            9 * 0.03 / (1 * 0.2 + 2 * 0.8 + 0.3 * 3 + 0.1 * 4 + 0.6 * 5 + 9),
        ]])
        np.testing.assert_allclose(albo_ref, albo)

        mult = expected_attention.mult_gate(log_unnormalized_attention_weights, self.independent.log_marginals())
        mult_ref = np.array([[
            1 * 0.2 / (1 * 0.2 + 2 * 0.8 + 0.3 * 3 + 0.1 * 4 + 0.6 * 5 + 0.1 * 6 + 0.8 * 7 + 0.07 * 8 + 0.03 * 9),
            2 * 0.8 / (1 * 0.2 + 2 * 0.8 + 0.3 * 3 + 0.1 * 4 + 0.6 * 5 + 0.1 * 6 + 0.8 * 7 + 0.07 * 8 + 0.03 * 9),
            3 * 0.3 / (1 * 0.2 + 2 * 0.8 + 0.3 * 3 + 0.1 * 4 + 0.6 * 5 + 0.1 * 6 + 0.8 * 7 + 0.07 * 8 + 0.03 * 9),
            4 * 0.1 / (1 * 0.2 + 2 * 0.8 + 0.3 * 3 + 0.1 * 4 + 0.6 * 5 + 0.1 * 6 + 0.8 * 7 + 0.07 * 8 + 0.03 * 9),
            5 * 0.6 / (1 * 0.2 + 2 * 0.8 + 0.3 * 3 + 0.1 * 4 + 0.6 * 5 + 0.1 * 6 + 0.8 * 7 + 0.07 * 8 + 0.03 * 9),
            6 * 0.1 / (1 * 0.2 + 2 * 0.8 + 0.3 * 3 + 0.1 * 4 + 0.6 * 5 + 0.1 * 6 + 0.8 * 7 + 0.07 * 8 + 0.03 * 9),
            7 * 0.8 / (1 * 0.2 + 2 * 0.8 + 0.3 * 3 + 0.1 * 4 + 0.6 * 5 + 0.1 * 6 + 0.8 * 7 + 0.07 * 8 + 0.03 * 9),
            8 * 0.07 / (1 * 0.2 + 2 * 0.8 + 0.3 * 3 + 0.1 * 4 + 0.6 * 5 + 0.1 * 6 + 0.8 * 7 + 0.07 * 8 + 0.03 * 9),
            9 * 0.03 / (1 * 0.2 + 2 * 0.8 + 0.3 * 3 + 0.1 * 4 + 0.6 * 5 + 0.1 * 6 + 0.8 * 7 + 0.07 * 8 + 0.03 * 9),
        ]])
        np.testing.assert_allclose(mult_ref, mult)
        sample_k = expected_attention.sample_k(log_unnormalized_attention_weights, self.independent.sample, 1000_000)
        np.testing.assert_allclose(ea_ref, sample_k, rtol=1e-2)

if __name__ == "__main__":
    # unittest.main()
    a = TestIndependent()
    a.setUp()
    a.test_independent()



