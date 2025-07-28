

import math
import unittest

import torch

from dblm.core.modeling import probability_tables, switching_tables


class TestSwitchingTableMessagepassing(unittest.TestCase):

    def setUp(self) -> None:
        self.switching_table = switching_tables.SwitchingTableForMessagePassing(5, [7,2,4,5,3])
        self.switching_table2 = switching_tables.SwitchingTableForMessagePassing(2, [3,4])

    def test_message_to_input_vars(self):
        # switch is 1 4 2 0 3 1 3
        # value is  3 1 2 4 0
        # offset is 0 7 9 13 18
        # so output is 8 18 11 3 17 8 17
        output_var = torch.tensor([[8,18,11,3,17,8,17], [8,18,11,3,17,8,17]])
        table = self.switching_table.message_to_var(0, [None] * 6 + [self.switching_table.output_var_assignment_to_message(output_var)]) #type:ignore
        reference = torch.ones((2,7,7), dtype=torch.float)
        reference[:, 3, :] = 0.0
        reference[:, 3, 3] = 1.0
        torch.testing.assert_close(table.potential_table(), reference) # type:ignore
        self.assertEqual(table.batch_size, (2,7)) # type:ignore

        table = self.switching_table.message_to_var(1, [None] * 6 + [self.switching_table.output_var_assignment_to_message(output_var)])   #type:ignore
        reference = torch.ones((2,7,2), dtype=torch.float)
        reference[:, 0, :] = 0.0
        reference[:, 0, 1] = 1.0
        reference[:, -2, :] = 0.0
        reference[:, -2, 1] = 1.0
        torch.testing.assert_close(table.potential_table(), reference) # type:ignore
        self.assertEqual(table.batch_size, (2,7)) # type:ignore

        table = self.switching_table.message_to_var(2, [None] * 6 + [self.switching_table.output_var_assignment_to_message(output_var)])   #type:ignore
        reference = torch.ones((2,7,4), dtype=torch.float)
        reference[:, 2, :] = 0.0
        reference[:, 2, 2] = 1.0
        torch.testing.assert_close(table.potential_table(), reference) # type:ignore
        self.assertEqual(table.batch_size, (2,7)) # type:ignore

    def test_message_to_switch_vars(self):
        # switch is 1 4 2 0 3 1 3
        # value is  3 1 2 4 0
        # offset is 0 7 9 13 18
        # so output is 8 18 11 3 17 8 17
        output_var = torch.tensor([[8,18,11,3,17,8,17], [8,18,11,3,17,8,17]])
        table = self.switching_table.message_to_var(self.switching_table.nvars - 2, [None] * 6 + [self.switching_table.output_var_assignment_to_message(output_var)])   #type:ignore
        reference = torch.zeros((2,7,5), dtype=torch.float)
        reference[:, 0, 1] = 1.0
        reference[:, 1, 4] = 1.0
        reference[:, 2, 2] = 1.0
        reference[:, 3, 0] = 1.0
        reference[:, 4, 3] = 1.0
        reference[:, 5, 1] = 1.0
        reference[:, 6, 3] = 1.0
        torch.testing.assert_close(table.potential_table(), reference) # type:ignore
        self.assertEqual(table.batch_size, (2,7)) # type:ignore

    def test_message_to_output_vars(self):
        input_logits = [
            torch.tensor([[1.0,2.0,3.0], [3.0,4.0,5.0]]),
            torch.tensor([[1.0,2.0,3.0, 4.0], [3.0,4.0,5.0, 6.0]]),
        ]
        input_messages = [
            probability_tables.LogLinearPotentialTable(
                table.size(), table, batch_dims=1
            )
            for table in input_logits
        ]
        switching_logits = torch.tensor([[1.0,1.0],[1.0,-math.inf]])
        switching_message = probability_tables.LogLinearPotentialTable(switching_logits.size(), switching_logits, batch_dims=1)
        message = self.switching_table2.message_to_var(self.switching_table2.nvars-1, [*input_messages, switching_message, None]) #type:ignore
        reference = torch.tensor([
            (input_logits[0][0].log_softmax(-1) + 1).tolist() + (input_logits[1][0].log_softmax(-1) + 1).tolist(),
            (input_logits[0][1].log_softmax(-1) + 1).tolist() + input_logits[1][1].log_softmax(-1).fill_(-math.inf).tolist(),
        ])
        torch.testing.assert_close(reference, message.log_potential_table())

if __name__ == "__main__":
    unittest.main()
