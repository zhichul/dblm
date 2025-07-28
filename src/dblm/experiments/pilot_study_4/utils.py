import math
import random
import numpy as np
import torch

from dblm.experiments.pilot_study_4 import distributions_pair


def add_uniform_noise(t, noise_level):
    t = torch.logaddexp(t + math.log(1-noise_level), t.new_zeros(t.size()).log_softmax(-1) + math.log(noise_level))
    return t

def encoder_decoder_train_data_generator(nvars, nvals, seq_len, n_branches, batch_size, x_model_seed):
    """Note that seq_len is the length WITHOUT [BOS]"""
    indices = distributions_pair.all_indices(seq_len, n_branches, list(range(nvars)), x_model_seed)
    indices_1 = torch.tensor([[i for i,j in index] for index in indices], dtype=torch.long)
    indices_2 = torch.tensor([[j for i,j in index] for index in indices], dtype=torch.long)
    ids = list(range(indices_1.size(0)))
    while True:
        z = torch.randint(nvals, (batch_size, nvars))
        z = z + (torch.arange(nvars) * nvals)[None,...]
        selection = random.sample(ids, batch_size)
        x_sel_1 = indices_1[selection]
        x_sel_2 = indices_2[selection]
        x_1 = torch.gather(z, 1, x_sel_1) # type:ignore
        x_2 = torch.gather(z, 1, x_sel_2) # type:ignore
        x = x_1 * nvars * nvals + x_2
        x = torch.cat([torch.empty((batch_size, 1), dtype=torch.long).fill_(nvars * nvals * nvars * nvals), x], dim=1) # prepend BOS
        xinput = x[..., :-1]
        xlabel = x[..., 1:]
        pz = torch.arange(nvars).expand_as(z)
        px = torch.arange(seq_len).expand(xinput.size())
        yield z, xinput, xlabel, pz, px


# def decoder_train_data_generator(nvars, nvals, seq_len, n_branches, batch_size, x_model_seed):
#     """Note that seq_len is the length WITHOUT [BOS]"""
#     indices = torch.tensor(distributions_pair.all_indices(seq_len, n_branches, list(range(nvars)), x_model_seed), dtype=torch.long)
#     ids = list(range(indices.size(0)))
#     while True:
#         z = torch.randint(nvals, (batch_size, nvars))
#         z = z + (torch.arange(nvars) * nvals)[None,...]
#         x_sel = indices[random.sample(ids, batch_size)]
#         x = torch.gather(z, 1, x_sel) # type:ignore
#         x = torch.cat([torch.empty((batch_size, 1), dtype=torch.long).fill_(nvars * nvals), x], dim=1) # prepend BOS
#         px = torch.arange(seq_len + 1).expand(x.size())
#         yield x, px