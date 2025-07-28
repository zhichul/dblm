import dataclasses
import json
import os
import random
import numpy as np
import torch
import tqdm
from transformers import BertForMaskedLM

from dblm.utils import seeding
from dblm.experiments.filter_ptb import filter_fn



out_dir="../spelling_data/temperature"
os.makedirs(out_dir, exist_ok=True)
nvars = 12
nvals = 30
length = 12
batch_size = 2048
mask_id=27
########
model = BertForMaskedLM.from_pretrained("../ptb/out/")

c2i = {c:i+1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
i2c = {i+1:c for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
i2c[0]="#"
i2c[27]="*"
i2c[28]="_"
c2i["#"]=0
c2i["*"]=27
c2i["_"]=28

def decode(lis):
    return "".join([i2c[i] for i in lis])

def encode(s):
    l = [0] * 12
    for i, c in enumerate(s[:12]):
        l[i] = c2i[c]
    return l


class State:

    def __init__(self, vec, n) -> None:
        self.vec = vec
        self.n = n
        self.index = self.encode(vec, n)

    @staticmethod
    def encode(vec, n):
        index = 0
        for i in vec:
            index *= n
            index += i
        return index

    @staticmethod
    def from_index(index, size, n):
        vec = []
        for _ in range(size):
            vec.append(index % n)
            index //= n
        return State(list(reversed(vec)), n)

    def __repr__(self) -> str:
        return f"State({self.vec}, {self.n})"

class MarkovChain:
    def __init__(self, states, initial:torch.Tensor, transition:torch.Tensor, seed) -> None:
        self.states = states
        self.initial = initial
        self.transition = transition
        self.seed = seed

    def sample(self, l=12,k=1):
        path = []
        prev = torch.distributions.Categorical(probs=self.initial).sample((k,)) #type:ignore
        path.append(prev[:,None])
        for i in range(l-1):
            transition = self.transition[prev]
            prev = torch.distributions.Categorical(probs=transition).sample((1,))[0] #type:ignore
            path.append(prev[:,None])
        return torch.cat(path, dim=1)

    def to(self, something):
        self.initial = self.initial.to(something)
        self.transition = self.initial.to(something)
        return self

    def path_to_state(self, paths):
        if isinstance(paths, torch.Tensor):
            if not len(paths.size()) == 2:
                raise ValueError(paths.size())
            paths = paths.tolist()
        state_seqs = []
        for path in paths:
            state_seq = [self.states[i] for i in path]
            state_seqs.append(state_seq)
        return state_seqs

    def state_dict(self):
        return dict(initial=self.initial, transition=self.transition, seed=self.seed)

    def load_state_dict(self, d):
        for k, v in d.item():
            setattr(self, k, v)

    @staticmethod
    def random_chain_with_symmetric_dirichlet(states, dirichlet_concentration=0.5, round=True, seed=12309):
        """Higher dirichlet concentration, the more it's concentrated in high entropy regions"""
        seeding.seed(seed)
        K = len(states)
        transition_matrix = [np.random.dirichlet([dirichlet_concentration] * K) for _ in range(K+1)]
        if round:
            for i in range(len(transition_matrix)):
                transition_matrix[i] = transition_matrix[i].round(2) # sparsify
                transition_matrix[i] = transition_matrix[i] / transition_matrix[i].sum() # renormalize
        transition_matrix = np.stack(transition_matrix)
        return MarkovChain(states, torch.from_numpy(transition_matrix[0]), torch.from_numpy(transition_matrix[1:]), seed)

debug = True
@torch.no_grad()
def generate(model, train_size, dev_size, test_size, chain: MarkovChain, nvars, nvals, length,mask_id, batch_size=16, temperatures=None):
    temperatures = torch.tensor(temperatures)
    train_zcat = []
    train_infcat = []
    train_xcat = []
    dev_zcat = []
    dev_infcat = []
    dev_xcat = []
    test_zcat = []
    test_infcat = []
    test_xcat = []
    for off in tqdm.tqdm(range(0, train_size * 2, batch_size)):
        xs = []
        paths = chain.sample(k=batch_size,l=length)
        # only for single marginals
        infs = []
        curr_masked_inputs = torch.zeros(batch_size, nvars, dtype=torch.long).fill_(mask_id)
        for i in range(paths.size(1)):
            logits = model(input_ids=curr_masked_inputs.to("cuda")).logits
            index = paths[:,i].unsqueeze(-1)
            observed_val = curr_masked_inputs.gather(-1, index) 
            unobserved = observed_val == mask_id
            tempered_logits = logits.gather(1, index.to("cuda").unsqueeze(-1).expand(logits.size(0), 1, logits.size(2))).squeeze(1) / temperatures[index.view(-1)].unsqueeze(-1).to("cuda")
            samples = torch.distributions.Categorical(logits=tempered_logits).sample((1,))[0].unsqueeze(-1).cpu() # type:ignore
            #force consistency
            samples = torch.where(unobserved, samples, observed_val)
            xs.append(index)
            xs.append(samples+nvars)
            curr_masked_inputs.scatter_(-1, index, samples)
            infs.append(logits.unsqueeze(1).cpu())
        logits = model(input_ids=curr_masked_inputs.to("cuda")).logits
        infs.append(logits.unsqueeze(1).cpu())
        infs = torch.cat(infs, dim=1).log_softmax(-1).reshape(batch_size, length+1, -1).repeat_interleave(2, -2)[...,:-1,:]
        xs = torch.cat(xs, dim=1)
        zs = curr_masked_inputs
        dev_inds = []
        test_inds = []
        train_inds = []
        for i, x in enumerate(xs.tolist()):
            hh = hash(tuple(x))
            if  hh% 4 < 2:
                train_inds.append(i)
            elif hh%4 == 2:
                dev_inds.append(i)
            else:
                test_inds.append(i)
        train_xcat.append(xs[train_inds])
        train_zcat.append(zs[train_inds])
        train_infcat.append(infs[train_inds])
        dev_xcat.append(xs[dev_inds])
        dev_zcat.append(zs[dev_inds])
        dev_infcat.append(infs[dev_inds])
        test_xcat.append(xs[test_inds])
        test_zcat.append(zs[test_inds])
        test_infcat.append(infs[test_inds])

    train_obj = {
        "x": torch.cat(train_xcat, dim=0),
        "z": torch.cat(train_zcat, dim=0),
        "inf": torch.cat(train_infcat, dim=0),
    }
    torch.save(train_obj,os.path.join(out_dir, "train.bin"))
    dev_obj = {
        "x": torch.cat(dev_xcat, dim=0)[:dev_size],
        "z": torch.cat(dev_zcat, dim=0)[:dev_size],
        "inf": torch.cat(dev_infcat, dim=0)[:dev_size],
    }
    torch.save(dev_obj,os.path.join(out_dir, "dev.bin"))
    test_obj = {
        "x": torch.cat(test_xcat, dim=0)[:test_size],
        "z": torch.cat(test_zcat, dim=0)[:test_size],
        "inf": torch.cat(test_infcat, dim=0)[:test_size],
    }
    torch.save(test_obj,os.path.join(out_dir, "test.bin"))
    return train_obj["x"].size(0), dev_obj["x"].size(0) + test_obj["x"].size(0), min(dev_obj["x"].size(0), dev_size), min(test_obj["x"].size(0), test_size)

def main():
    seeding.seed(12309)
    temperatures = [0.3 + random.random() * 1.4 for _ in range(nvars)]
    dirichlet_concentration = 0.7
    model.to("cuda") # type:ignore

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(dict(
            z_entropy=0),f)

    states = [State([i], nvars) for i in range(nvars)]
    chain = MarkovChain.random_chain_with_symmetric_dirichlet(states, dirichlet_concentration=dirichlet_concentration)
    torch.save(chain.state_dict(), os.path.join(out_dir, "chain.bin"))

    train_size, eval_size, dev_size, test_size = generate(model, 400000, 12800, 12800, chain=chain, nvars=nvars, nvals=nvals, length=length, mask_id=mask_id, batch_size=batch_size,temperatures=temperatures)
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(dict(
            out_dir=out_dir,
            nvars = nvars,
            nvals = nvals,
            seq_len = length * 2,
            batch_size = batch_size,
            mask_id=mask_id,
            dirichlet_concentration=dirichlet_concentration,
            encoder_vocab_size=30 + 1,
            decoder_vocab_size=30 + 12 + 1,
            encoder_pad_token_id=30,
            decoder_bos_token_id=30+12,
            dev_size=dev_size,
            test_size=test_size,
            eval_size=eval_size,
            train_size=train_size,
            temperatures=temperatures
        ),f)


def test():
    n = 12
    size = 2
    assert State.from_index(143, size, n).vec == [11, 11]
    assert State.from_index(143, size, n).n == 12
    assert State((11,11), 12).index == 143

if __name__ == "__main__":

    test()
    main()





    # while True:
    #     items = input("Enter a word, will be padded to 12 or truncated...\n").split()
    #     if len(items) == 1:
    #         s = items[0]
    #         temp = 1.0
    #     elif len(items) == 2:
    #         temp = float(items[1])
    #         s = items[0]
    #     else:
    #         print("bad format, continuing")
    #         continue
    #     input_ids = torch.tensor(encode(s)).unsqueeze(0)
    #     logits = model(input_ids).logits[0] # type:ignore
    #     sample = torch.distributions.Categorical(logits=logits / temp).sample((1,))[0] # type:ignore
    #     is_mask = input_ids == 27
    #     output = torch.where(is_mask, sample, input_ids)
    #     print(decode(output.view(-1).tolist()))
