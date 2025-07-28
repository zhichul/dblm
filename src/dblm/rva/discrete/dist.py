
import numpy as np


class Distribution:

    def __init__(self, nvars, nvals):
        self.nvars = nvars
        self.nvals = nvals
        self.nv = sum(nvals)
        self.offsets = [sum(self.nvals[:i]) for i in range(self.nvars)]

    def bin_to_int(self, mat):
        if not len(mat.shape) == 2 and mat.shape[1] == self.nv:
            raise ValueError(mat.shape, self.nv)
        cols = []
        for i in range(self.nvars):
            cols_bin = mat[:, self.offsets[i]:self.offsets[i] + self.nvals[i]]
            col_int = np.argmax(cols_bin, axis=1)
            cols.append(col_int[:, None])
        return np.concatenate(cols, axis=1)

    def int_to_bin(self, mat):
        if not len(mat.shape) == 2 and mat.shape[1] == self.nvars:
            raise ValueError(mat.shape, self.nvars)
        cols = []
        for i in range(self.nvars):
            cols_bin = np.zeros((mat.shape[0], self.nvals[i]))
            cols_bin[np.arange(mat.shape[0]), mat[:,i]] = 1
            cols.append(cols_bin)
        return np.concatenate(cols, axis=1)

    def start_end(self, i):
        return self.offsets[i], self.offsets[i] + self.nvals[i]
