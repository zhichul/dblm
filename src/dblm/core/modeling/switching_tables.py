from __future__ import annotations
import code
from typing import Sequence
from dblm.core.interfaces import pgm
from dblm.core.modeling import constants, probability_tables, utils
import torch.nn as nn
import torch
import math


class SwitchingTable(probability_tables.LogLinearProbabilityMixin, probability_tables.LogLinearTableInferenceMixin, nn.Module, pgm.ProbabilityTable):
    """This describes a potential table over nvars variables
    each with nvals[i] values, plus a single switch, and
    a output variable that is conceptually a pair (selected_of_nvar, val_of_selected)
    encoded as a single discrete value, since the nvars may not
    have the same number of possible values.

    Variable is indexed as: [0:(nvars-1) is z0 |nvars is switch |nvars+1 is output]
    """

    def __init__(self, nvars: int, nvals: list[int], mode:constants.SwitchingMode = constants.SwitchingMode.VARPAIRVAL) -> None:
        self.call_super_init = True
        if mode == constants.SwitchingMode.VARPAIRVAL:
            super().__init__(nvars + 2, list(range(nvars + 1))) # parents
            self._nvals = [*nvals, nvars, sum(nvals)] # [nvars] is for the switching variable, sum(nvals) is for the output variable
            self._nvars = nvars + 2 # 1 is for the switching variable, 1 for the output variable
            logits = torch.zeros(self._nvals).fill_(-math.inf)
            colon = slice(None, None, None)
            for sw in range(nvars):
                for val in range(nvals[sw]):
                    # this line indexes the rows of the table whose sw'th variable takes value val
                    logits[(colon,) * sw + (val,) + (colon,) * (nvars - sw - 1) + (sw, sum(nvals[:sw]) + val)] = 0
            self.logits = nn.Parameter(logits, requires_grad=False)
        elif mode == constants.SwitchingMode.MIXTURE:
            if not len(set(nvals)) == 1:
                raise ValueError(f"Cannot have mixture of different outcome space sizes: {nvals}")
            super().__init__(nvars + 2, list(range(nvars + 1))) # parents
            self._nvars = nvars + 2 # 1 is for the switching variable, 1 for the output variable
            self._nvals = [*nvals, nvars, nvals[0]] # [nvars] is for the switching variable, nvals[0] is for the output variable
            logits = torch.zeros(self._nvals).fill_(-math.inf)
            colon = slice(None, None, None)
            for sw in range(nvars):
                for val in range(nvals[sw]):
                    # this line indexes the rows of the table whose sw'th variable takes value val
                    logits[(colon,) * sw + (val,) + (colon,) * (nvars - sw - 1) + (sw, val)] = 0
            self.logits = nn.Parameter(logits, requires_grad=False)
        else:
            raise NotImplementedError()
        self.mode = mode

    def probability_table(self) -> torch.Tensor:
        return self.logits.exp() # type:ignore

    def log_probability_table(self) -> torch.Tensor:
        return self.logits # type:ignore

    def __repr__(self):
        return f"SwitchingTable({self.mode})"

    def expand_batch_dimensions_(self, batch_sizes: tuple[int, ...]) -> SwitchingTable:
        self.expand_batch_dimensions_meta_(batch_sizes) # type:ignore
        self.logits.data = self.logits.data.expand((*self.batch_size, *self.nvals))
        return self

    def expand_batch_dimensions(self, batch_sizes: tuple[int, ...]) -> SwitchingTable:
        # make a copy then do it in place
        table = SwitchingTable(self.nvars-2, self.nvals[:self.nvars-2], mode=self.mode)
        table.expand_batch_dimensions_(batch_sizes + self.batch_size) # type:ignore
        table.logits.data = table.logits.data.to(self.logits.data.device)
        return table

    def condition_on(self, observation: dict[int, int] | dict[int, torch.Tensor], cartesian=True, bp=False):
        if bp:
            if cartesian: raise ValueError("Cartesian=True not supported in bp mode.")
            if self.mode == constants.SwitchingMode.VARPAIRVAL:
                 # type:ignore
                bp_table = self.to_bp_table()
                bp_table.condition_on(observation)
                if isinstance(next(observation.values().__iter__()), torch.Tensor):
                    bp_table.expand_batch_dimensions_(next(observation.values().__iter__()).size()) # type:ignore
                return bp_table
            else:
                raise NotImplementedError()
        else:
            table = super().condition_on(observation, cartesian)
            table.logits.data = table.logits.data.to(self.logits.data.device)
            return table

    def to_bp_table(self):
        if self.mode == constants.SwitchingMode.VARPAIRVAL:
            bp_table = SwitchingTableForMessagePassing(self.nvars-2, self.nvals[:-2], mode=self.mode).to(self.logits.device)
        else:
            bp_table = self
        return bp_table

class SwitchingTableForMessagePassing(probability_tables.LogLinearProbabilityMixin, nn.Module, pgm.ProbabilityTable):
    """This describes a potential table over nvars variables
    each with nvals[i] values, plus a single switch, and
    a output variable that is conceptually a pair (selected_of_nvar, val_of_selected)
    encoded as a single discrete value, since the nvars may not
    have the same number of possible values.

    Variable is indexed as: [0:(nvars-1) is z0 |nvars is switch |nvars+1 is output]
    """
    def __init__(self, nvars: int, nvals: list[int], mode:constants.SwitchingMode = constants.SwitchingMode.VARPAIRVAL) -> None:
        self.call_super_init = True
        if mode == constants.SwitchingMode.VARPAIRVAL:
            super().__init__(nvars + 2, list(range(nvars + 1))) # parents
            self._nvals = [*nvals, nvars, sum(nvals)] # [nvars] is for the switching variable, sum(nvals) is for the output variable
            self._nvars = nvars + 2 # 1 is for the switching variable, 1 for the output variable
            self._output_var_to_switch = [switch_val for switch_val, nval in enumerate(nvals) for _ in range(nval)]
            self.register_buffer("_output_var_to_switch_tensor", torch.tensor([switch_val for switch_val, nval in enumerate(nvals) for _ in range(nval)]))
            self._output_var_to_val = [val for _, nval in enumerate(nvals) for val in range(nval)]
            self.register_buffer("_output_var_to_val_tensor", torch.tensor([val for _, nval in enumerate(nvals) for val in range(nval)]))
        else:
            raise NotImplementedError()
        self.observation = None
        self.mode = mode

    @property
    def device(self):
        return self._output_var_to_switch_tensor.device

    @property
    def logits(self):
        raise NotImplementedError("SwitchingTableForMessagePassing does not define logits.")

    def probability_table(self) -> torch.Tensor:
        raise NotImplementedError("SwitchingTableForMessagePassing is not materialized and thus cannot return prob table.")

    def log_probability_table(self) -> torch.Tensor:
        raise NotImplementedError("SwitchingTableForMessagePassing is not materialized and thus cannot return logprob table.")

    def __repr__(self):
        return f"SwitchingTableForMessagePassing({self.mode})"

    def expand_batch_dimensions_(self, batch_sizes: tuple[int, ...]) -> SwitchingTableForMessagePassing:
        self.expand_batch_dimensions_meta_(batch_sizes) # type:ignore
        return self

    def expand_batch_dimensions(self, batch_sizes: tuple[int, ...]) -> SwitchingTableForMessagePassing:
        # make a copy then do it in place
        table = SwitchingTableForMessagePassing(self.nvars-2, self.nvals[:self.nvars-2], mode=self.mode).to(device=self.device) # type:ignore
        table.expand_batch_dimensions_(batch_sizes + self.batch_size) # type:ignore
        return table

    def marginalize_over(self, variables: Sequence[int]):
        raise NotImplementedError("SwitchingTableForMessagePassing only supports messages and does not materialize table for marginalization.")

    def condition_on(self, observation: dict[int, int] | dict[int, torch.Tensor], cartesian=True):
        if self.observation is None:
            self.observation = observation
            return self
        else:
            raise ValueError("SwitchingTableForMessagePassing can only be conditioned on once.")

    def message_to_var(self, i, messages_other: Sequence[pgm.PotentialTable]):
        if self.observation is None and all(m is None for m in messages_other):
            return None
        else:
            messages_full = [None] * self.nvars
            idx = 0
            for j in range(self.nvars):
                if self.observation and j in self.observation: # type:ignore
                    table = utils.index_to_one_hot(torch.tensor(self.observation[j]) if not isinstance(self.observation[j], torch.Tensor) else self.observation[j], self.nvals[j], device=self.device) # type:ignore
                    messages_full[j] = probability_tables.LogLinearPotentialTable(table.size(), table, batch_dims=self.batch_dims) # type:ignore
                else:
                    msg = messages_other[idx]
                    if msg is None and j != i:
                        table = torch.zeros((*self.batch_size, self.nvals[j]), device=self.device) # type:ignore
                        msg = probability_tables.LogLinearPotentialTable(table.size(), table, batch_dims=self.batch_dims) # type:ignore
                    messages_full[j] = msg # type:ignore
                    idx += 1
            messages_other = messages_full # type:ignore
            if i < self.nvars - 2:
                return self._message_to_input_vars(i, messages_other[-1])
            elif i == self.nvars - 2:
                return self._message_to_switching_vars(messages_other[-1])
            else:
                return self._message_to_output_vars(messages_other[:self.nvars-2], messages_other[self.nvars-2])

    def _message_to_input_vars(self, input_var_index: int, output_var_message: pgm.PotentialTable):
        #TODO WHY TF ARE WE NOT MULTIPLYING IN THE SWITCH VARIABLE DISTRIBUTIONS?
        if output_var_message is None:
            return None
        if self.mode == constants.SwitchingMode.VARPAIRVAL:
            n = self.nvals[input_var_index]
            offset = sum(self.nvals[:input_var_index])
            input_message_logits =  output_var_message.log_potential_table()[..., offset:offset + n]
            input_message_logits[(input_message_logits == -math.inf).all(dim=-1, keepdim=True).expand_as(input_message_logits)] = 0.0
            return probability_tables.LogLinearPotentialTable(input_message_logits.size(), input_message_logits, batch_dims=output_var_message.batch_dims)
        else:
            raise ValueError(f"unknown switching mode {self.mode}")

    def _message_to_switching_vars(self, output_var_message: pgm.PotentialTable):
        if output_var_message is None:
            return None
        if self.mode == constants.SwitchingMode.VARPAIRVAL:
            output_logits = output_var_message.log_potential_table()
            offset = 0
            message = []
            for nval in self.nvals[:self.nvars - 2]:
                message.append(torch.logsumexp(output_logits[...,offset:offset +  nval], dim=-1, keepdim=True))
                offset += nval
            message = torch.cat(message, dim=-1)
            return probability_tables.LogLinearPotentialTable(message.size(), message, batch_dims=output_var_message.batch_dims)
        else:
            raise ValueError(f"unknown switching mode {self.mode}")

    def _message_to_output_vars(self, input_messages, switching_message):
        # only works with messages instead of one-hot variable assignments, since we are passing to the output_var
        # normalize since we are manually doing mixture
        input_messages = [input_message.renormalize() if not isinstance(input_message, pgm.ProbabilityTable) else input_message for input_message in input_messages]
        input_logits = [input_message.log_potential_table() for input_message in input_messages]
        switch_logits = switching_message.log_potential_table()
        batch_size = switching_message.batch_size
        output_logits = torch.zeros((*batch_size, self.nvals[-1]), device=self.device).fill_(-math.inf) # type:ignore
        offset = 0
        for input_var_index, (nval, input_logit) in enumerate(zip(self.nvals, input_logits)): # self.nvals should have 2 more items than input_logits
            output_logits[(slice(None),) * len(batch_size) + (slice(offset, offset+nval),)] = input_logit + switch_logits[..., input_var_index:input_var_index+1]
            offset += nval
        return probability_tables.LogLinearPotentialTable(output_logits.size(), output_logits, batch_dims=len(batch_size))

    def output_var_assignment_to_message(self, assignment: torch.Tensor):
        return probability_tables.index_to_potential_table_message(assignment, self.nvals[-1])


