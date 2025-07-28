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



out_dir="../spelling_data/default"
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
def generate(model, file, name, chain: MarkovChain, nvars, nvals, length,mask_id, batch_size=16, indices=None):
    arr = np.memmap(name, dtype=np.uint8, mode='r').reshape(-1, nvars)
    zcat = []
    infcat = []
    xcat = []
    if indices is not None:
        arr = arr[indices]
    for off in tqdm.tqdm(range(0, arr.shape[0], batch_size)):
        zs = torch.from_numpy(arr[off:off + batch_size]).to(torch.long)
        paths = chain.sample(k=zs.size(0),l=length)
        index_states = chain.path_to_state(paths)
        xs = []
        inf_masked_inputs = []

        for i, index_state_seq in enumerate(index_states):
            # make x
            x = []
            for s in index_state_seq:
                x.append(s.index)
                x.append(nvars + State.encode([zs[i, index].item() for index in s.vec], nvals))
            xs.append(x)
            # make inf input
            inf_masked_inputs_i = zs[i,None,:].expand(length+1, zs.size(1)).clone().fill_(mask_id)
            for j in range(length+1):
                observed = sum((s.vec for s in index_state_seq[:j]), [])
                inf_masked_inputs_i[j, observed] = zs[i, observed]
            inf_masked_inputs.append(inf_masked_inputs_i)
        inf_masked_inputs = torch.stack(inf_masked_inputs)
        xcat.append(torch.tensor(xs))
        zcat.append(zs)
        logits = model(input_ids=inf_masked_inputs.reshape(-1, nvars).to("cuda")).logits.reshape(zs.size(0), length+1, nvars, -1)
        infcat.append(logits.log_softmax(-1).reshape(zs.size(0), length+1, -1).repeat_interleave(2, -2)[...,:-1,:].cpu())
    torch.save({
        "x": torch.cat(xcat, dim=0),
        "z": torch.cat(zcat, dim=0),
        "inf": torch.cat(infcat, dim=0),
    },file)

def main():
    dirichlet_concentration = 0.7
    model.to("cuda") # type:ignore

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(dict(
            z_entropy=0),f)

    states = [State([i], nvars) for i in range(nvars)]
    chain = MarkovChain.random_chain_with_symmetric_dirichlet(states, dirichlet_concentration=dirichlet_concentration)
    torch.save(chain.state_dict(), os.path.join(out_dir, "chain.bin"))

    train_indices = []
    evaluation_indices = []
    with open("../ptb/ptb.train.txt", "rt") as f:
        tokens = [token for line in f for token in filter(filter_fn, line.strip().split())]
        for i in range(len(tokens)):
            if hash(tokens[i]) % 2 == 0:
                train_indices.append(i)
            else:
                evaluation_indices.append(i)

    seeding.seed(12093)
    random.shuffle(evaluation_indices)
    validation_indices = evaluation_indices[:len(evaluation_indices)//2][:12800]
    test_indices = evaluation_indices[len(evaluation_indices)//2:][:12800]
    with open(os.path.join(out_dir, "big_args.json"), "w") as f:
        json.dump(dict(
            train_indices=train_indices,
            evaluation_indices=evaluation_indices,
            validation_indices=validation_indices,
            test_indices=test_indices,
        ),f)
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
            dev_size=len(validation_indices),
            test_size=len(test_indices),
            eval_size=len(evaluation_indices),
            train_size=len(train_indices),
        ),f)
    generate(model, os.path.join(out_dir, "dev.bin"), "../ptb/train.bin", chain=chain, nvars=nvars, nvals=nvals, length=length, mask_id=mask_id, batch_size=batch_size, indices=validation_indices)
    generate(model, os.path.join(out_dir, "test.bin"), "../ptb/train.bin", chain=chain, nvars=nvars, nvals=nvals, length=length, mask_id=mask_id, batch_size=batch_size, indices=test_indices)
    generate(model, os.path.join(out_dir, "train.bin"), "../ptb/train.bin", chain=chain, nvars=nvars, nvals=nvals, length=length, mask_id=mask_id, batch_size=batch_size, indices=train_indices)





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
