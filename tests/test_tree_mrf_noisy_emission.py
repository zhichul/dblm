
import os
import unittest

import torch
from dblm.core.modeling import constants

from dblm.experiments.pilot_study_1 import tree_mrf_noisy_emission


class TestTreeMRFNoisyEmission(unittest.TestCase):

    def setUp(self) -> None:
        torch.set_num_threads(12)
        self.model = tree_mrf_noisy_emission.TreeMrfNoisyEmission(
            8, 5, constants.GaussianInitializer(mean=0, std=10, min=-20, max=20), (4,1), # type:ignore
            5, constants.GaussianInitializer(mean=0, std=10, min=-20, max=20), (4,1)) # type:ignore
        self.save_dir = "/tmp/TestTreeMRFNoisyEmission"
        os.system(f"rm -rf {self.save_dir}")
        os.makedirs(self.save_dir, exist_ok=False)
        self.model.save(self.save_dir)
        self.addCleanup(lambda: os.system(f"rm -rf {self.save_dir}"))

    def test_save_load(self):
        self.loaded_model = tree_mrf_noisy_emission.TreeMrfNoisyEmission.load(self.save_dir)
        assignment = list(zip([132, 137, 142, 147, 152], [0, 1, 2, 3, 4]))
        loaded_inference = self.loaded_model.nested_model.log_marginal_probability(assignment=assignment, iterations=5)
        original_inference = self.model.nested_model.log_marginal_probability(assignment=assignment, iterations=5)
        torch.testing.assert_close(loaded_inference, original_inference)

if __name__ == "__main__":
    unittest.main()
