from __future__ import annotations
from dblm.core.modeling import factor_graphs
from dblm.core.modeling.switching_tables import SwitchingTable


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
