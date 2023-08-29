from __future__ import annotations
from dblm.core.interfaces import pgm
from dblm.core.modeling import constants, factor_graphs, probability_tables
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
            _nvals = [*nvals, nvars, sum(nvals)] # [nvars] is for the switching variable, sum(nvals) is for the output variable
            super().__init__(_nvals, list(range(nvars + 1))) # size and parents
            self._nvars = nvars + 2 # 1 is for the switching variable, 1 for the output variable
            self._nvals = _nvals
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
            _nvals = [*nvals, nvars, nvals[0]] # [nvars] is for the switching variable, nvals[0] is for the output variable
            super().__init__(_nvals, list(range(nvars + 1))) # size and parents
            self._nvars = nvars + 2 # 1 is for the switching variable, 1 for the output variable
            self._nvals = _nvals
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
        return self.logits.exp().expand(*(*self.batch_size, *self.nvals)) # type:ignore

    def log_probability_table(self) -> torch.Tensor:
        return self.logits.expand(*(*self.batch_size, *self.nvals)) # type:ignore

    def __repr__(self):
        return f"SwitchingTable({self.mode})"

    def expand_batch_dimensions(self, batch_sizes: tuple[int, ...]) -> SwitchingTable:
        return super().expand_batch_dimensions(batch_sizes) # type:ignore

class BatchedSwitchingTables(factor_graphs.FactorGraph):

    def __init__(self, nvars: int, nvals: int | list[int], nswitches: int) -> None:
        """This instantiates a series of `nswtiches` SwitchingTables.
        Every switch variable has its own factor that happens to be the
        SAME deterministic function.

        Variable is indexed as: [0:(nvars-1) is z0
                                nvars:nvars+nswitches-1 are switches
                                nvars+nswitches:nvars+2*nswitches-1 are outputs]
        """
        if isinstance(nvals, int):
            nvals = [nvals] * nvars
        switching_table = SwitchingTable(nvars, nvals)
        super().__init__(nvars + nswitches + nswitches,
                        nvals + [nvars] * nswitches + [sum(nvals)] * nswitches,
                        [(*tuple(range(nvars)), nvars + i, nvars + nswitches + i) for i in range(nswitches)],
                        [switching_table for _ in range(nswitches)])
