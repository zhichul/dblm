
import unittest

from dblm.utils import seeding
from dblm.rva import unnormalized_attention
import numpy as np

class TestUnnormalizedAttention(unittest.TestCase):

    def setUp(self) -> None:
        self.intravar = unnormalized_attention.IntraVar(
            nvars=3,
            nvals=np.array([2,3,4]),
            scales_mu=np.array([1,2,3.2]),
            scales_d=np.array([0.1,0.3,0.4]),
            corr_d=[np.eye(2), np.eye(3), np.eye(4)]
        )
        self.intervar = unnormalized_attention.InterVar(
            nvars=3,
            nvals=np.array([2,3,4]),
            scales_mu=np.array([1,2,3]),
            corr_mu=np.array(
                [
                    [1., 0.86, 0.21],
                    [0.86, 1., 0.35],
                    [0.21, 0.35, 1.]
                ]
            ),
            scales_d=np.array([0.1,0.3,0.4]),
            corr_d=[np.eye(2), np.eye(3), np.eye(4)]
        )

    def test_intra_var(self):
        seeding.seed(190)
        N = 10_000_000
        X = self.intravar.sample(N)
        cov = (X.transpose() @ X) / N - X.mean(axis=0)[:,None] @ X.mean(axis=0)[None,:]
        print(cov.round(2))

    def test_inter_var(self):
        seeding.seed(190)
        N = 10_000_000
        X = self.intervar.sample(N)
        cov = (X.transpose() @ X) / N - X.mean(axis=0)[:,None] @ X.mean(axis=0)[None,:]
        print(cov.round(2))

if __name__ == "__main__":
    unittest.main()
    # a = TestUnnormalizedAttention()
    # a.setUp()
    # a.test_intra_var()
    # a.test_inter_var()



