import unittest

import torch

from dblm.core.modeling import constants, probability_tables


class TestTensorTables(unittest.TestCase):

    def setUp(self) -> None:
        # create a table
        table = probability_tables.LogLinearPotentialTable((3,4,2), constants.TensorInitializer.CONSTANT)
        table._logits.data = torch.tensor([
            [[1.,2],
             [3,4],
             [5,6],
             [7,8]],
            [[9,10],
             [11,12],
             [13,14],
             [15,16]],
            [[17,18],
             [19,20],
             [21,22],
             [23,24]],
        ]).log()
        self.potential_table = table
        table = probability_tables.LogLinearProbabilityTable((3,4,2), [0], constants.TensorInitializer.CONSTANT)
        table._logits.data = torch.tensor([
            [[1.,2],
             [3,4],
             [5,6],
             [7,8]],
            [[9,10],
             [11,12],
             [13,14],
             [15,16]],
            [[17,18],
             [19,20],
             [21,22],
             [23,24]],
        ]).log()
        self.probability_table = table

    def test_potential_table_condition_on(self):
        # single batch conditioning
        cpt = self.potential_table.condition_on({0: torch.tensor([1,2]),
                                 1: torch.tensor([3,1])})
        torch.testing.assert_close(cpt.potential_table(),
                torch.tensor(
                    [[15., 16],
                     [19, 20]]
                ))
        # sequence of batch conditioning (treated as cartesian product)
        cpt = self.potential_table.condition_on({0: torch.tensor([1,2])}).condition_on({0: torch.tensor([3,1])})
        torch.testing.assert_close(cpt.potential_table(),
                torch.tensor(
                    [[[15., 16], # 1,3
                     [11, 12]],  # 1,1
                     [[23., 24], # 2,3
                     [19, 20]]]  # 2,1
                ))

    def test_probability_table_condition_on(self):
        # single batch conditioning
        cpt = self.probability_table.condition_on({0: torch.tensor([1,2]),
                                 1: torch.tensor([3,1])})
        # cpt is really a probability table since we only conditioned on one children (v0 par v1,v2 child),
        # but ProbabilityTable.potential_table is assumed to just return the same as
        # that by Probability.probability_table
        torch.testing.assert_close(cpt.potential_table(),
                torch.tensor(
                    [[15., 16],
                     [19, 20]]
                ).log().softmax(dim=-1))
        # sequence of batch conditioning on v0 then v1 (treated as cartesian product) note the indices are both 0
        # this is because after conditioning the first time the variables remaining are v1 and v2 so their
        # indices get shifted back to zero v1 -> renamed v0 v2 -> renamed v1
        cpt = self.probability_table.condition_on({0: torch.tensor([1,2])}).condition_on({0: torch.tensor([3,1])})
        torch.testing.assert_close(cpt.potential_table(),
                torch.tensor(
                    [[[15., 16], # 1,3
                     [11, 12]],  # 1,1
                     [[23., 24], # 2,3
                     [19, 20]]]  # 2,1
                ).log().softmax(dim=-1))

    def test_potential_table_marginalize(self):
        # single batch conditioning on v0 followed by marginalization on v1
        cpt = self.potential_table.condition_on({0: torch.tensor([1,2,1])})
        torch.testing.assert_close(cpt.potential_table(),
                torch.tensor(
                    [[[9.,10], # 1
                    [11,12],
                    [13,14],
                    [15,16]],
                    [[17,18], # 2
                    [19,20],
                    [21,22],
                    [23,24]],
                    [[9.,10], # 3
                    [11,12],
                    [13,14],
                    [15,16]]]
                ))
        cpt = cpt.marginalize_over([0]) # marginalize over v1
        torch.testing.assert_close(cpt.potential_table(),
                torch.tensor(
                    [[[9.,10], # 1
                    [11,12],
                    [13,14],
                    [15,16]],
                    [[17,18], # 2
                    [19,20],
                    [21,22],
                    [23,24]],
                    [[9.,10], # 3
                    [11,12],
                    [13,14],
                    [15,16]]]
                ).sum(dim=1))

    def test_probability_table_marginalize(self):
        # single batch conditioning on v0 followed by marginalization on v1
        cpt = self.probability_table.condition_on({0: torch.tensor([1,2,1])})
        torch.testing.assert_close(cpt.potential_table(),
                torch.tensor(
                    [[[9.,10], # 1
                    [11,12],
                    [13,14],
                    [15,16]],
                    [[17,18], # 2
                    [19,20],
                    [21,22],
                    [23,24]],
                    [[9.,10], # 3
                    [11,12],
                    [13,14],
                    [15,16]]]
                ).log().reshape(3,8).softmax(dim=-1).reshape(3,4,2))
        cpt = cpt.marginalize_over([0]) # marginalize over v1
        torch.testing.assert_close(cpt.potential_table(),
                torch.tensor(
                    [[[9.,10], # 1
                    [11,12],
                    [13,14],
                    [15,16]],
                    [[17,18], # 2
                    [19,20],
                    [21,22],
                    [23,24]],
                    [[9.,10], # 3
                    [11,12],
                    [13,14],
                    [15,16]]]
                ).log().reshape(3,8).softmax(dim=-1).reshape(3,4,2).sum(dim=1))

    def test_potential_value(self):
        values = self.potential_table.expand_batch_dimensions((3,)).potential_value((torch.tensor([0,1,1]), torch.tensor([1,2,3]), torch.tensor([0,1,1]))) # type:ignore
        torch.testing.assert_close(values, torch.tensor([3,14,16.]))
        values = self.probability_table.expand_batch_dimensions((3,)).potential_value((torch.tensor([0,1,1]), torch.tensor([1,2,3]), torch.tensor([0,1,1]))) # type:ignore
        torch.testing.assert_close(values, torch.tensor([3/36,0.14,0.16]))

if __name__ == "__main__":
    unittest.main()
