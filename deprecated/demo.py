from itertools import product

import numpy as np
import torch
from matplotlib import pyplot as plt

from experiments.utils.seeding import seed
from nilm import random_graph

def generate_pgm(nvars=5, nvals=4, mode="random_spanning_tree"):
    if mode == "random_spanning_tree":
        rg = random_graph.RandomGraph()
        vertices = rg.run(nvars)
        edges = []
        for vertex in vertices:
            if vertex.pi is not None:
                edges.append((vertex.pi.index, vertex.index))
        log_potentials = np.random.rand(len(edges), nvals, nvals) #TODO: do a distribution such as gaussian
    return {
        "nodes": vertices,
        "edges": edges,
        "factors": edges,
        "log_potentials": log_potentials,
        "nvars": nvars,
        "nvals": nvals
    }

def materialize(pgm):
    table = []
    nvals = pgm["nvals"]
    nvars = pgm["nvars"]
    factors = pgm["factors"]
    log_potentials = pgm["log_potentials"]
    for row in product(*([list(range(nvals))] * nvars)):
        log_potential = 0
        for i, factor in enumerate(factors):
            value = [row[v] for v in factor]
            log_potential += log_potentials[tuple([i] + value)]
        table.append(log_potential)
    table = np.array(table).reshape(*([nvals] * nvars))
    return table

def entropy(table):
    table = torch.tensor(table)
    dist = torch.distributions.categorical.Categorical(logits=table.view(-1))
    return dist.entropy().numpy()

def pgm_sampler(table, nvars, nvals):
    table = torch.tensor(table)
    assert table.numel() == nvals ** nvars
    dist = torch.distributions.categorical.Categorical(logits=table.view(-1))
    def sample(n):
        samples = dist.sample(sample_shape=(n,)).numpy()
        samples = np.stack([(samples // (nvals ** (nvars-1-i))) % nvals for i in range(nvars)],axis=1)
        return samples
    return sample


###############
def generate_markov_chain(nstates=5, transition_mode="full"):
    if transition_mode == "full":
        # row-stochastic matrix
        transition = torch.softmax(torch.tensor(np.random.rand(nstates, nstates)),dim=1).numpy()
        initial = torch.softmax(torch.tensor(np.random.rand(1,nstates)), dim=0).numpy()
    return {
        "transition": transition,
        "initial": initial
    }

def chain_sampler(markov_chain):
    transition = markov_chain["transition"]
    initial = markov_chain["initial"]
    transition = torch.distributions.categorical.Categorical(logits=torch.tensor(transition))
    initial = torch.distributions.categorical.Categorical(logits=torch.tensor(initial))
    def sample(n, t):
        state = initial.sample((n,)).reshape(-1)
        states = [state]
        for i in range(t-1):
            transitions = transition.sample((n,))
            state = torch.gather(transitions, -1, state[:, None]).reshape(-1)
            states.append(state)
        return torch.stack(states).numpy()
    return sample


#################
class UniformlyNoisyEmission:

    # only works with discrete pgms where each variable can take on
    # the same number of values

    def __init__(self, nstate, nvals, noise):
        self.noise = noise
        self.vocab_size = nstate * nvals
        self.nstate = nstate
        self.nvals = nvals

    def __getitem__(self, var_index):
        def emission(value):
            out = np.zeros((self.vocab_size,)) * self.noise / (self.vocab_size - 1)
            out[var_index * self.nvals + value] = 1 - self.noise
            return out
        return emission

# we do the emission separately because it does not depend only on the hmm state
# but also z0
def generate_language_model(nstate=5, nvals=4, mode="uniformly_noisy", noise=0.1):
    if mode=="uniformly_noisy":
        return UniformlyNoisyEmission(nstate, nvals, noise)

def test_random_graph():
    rg = random_graph.RandomGraph()
    results = []
    for i in range(10000):
        vertices = rg.run(10)
        results.append(vertices[1].pi.index)
    plt.hist(results)
    print(results)
    plt.show()

def verbalize(sentence, nvars=5, nvals=4):
    return [("var%d=%d" % (token // nvals, token % nvals)) for token in sentence]

if __name__ == "__main__":
    seed(42)
    # test_random_graph()
    nvars = 5       # number of variables
    nvals = 4       # number of values per variable
    noise = 0.0     # noise when emitting xi given z0 and zi
    N = 10          # number of data points to generate
    T = 10          # length of each data point

    # randomly generate a tree structured MRF and sample from it brute force
    # materializing the probability table
    pgm = generate_pgm(nvars=nvars, nvals=nvals)
    table = materialize(pgm)
    ent = entropy(table)
    vsampler = pgm_sampler(table, nvars, nvals)
    vsamples = vsampler(N)
    print(ent)
    print(vsamples)

    # randomly sample a markov chain for z>0 and sample from it by ancestral sampling
    chain = generate_markov_chain(nstates=nvars)
    ssampler = chain_sampler(chain)
    ssamples = ssampler(N, T)
    print(ssamples)

    # emission distribution has 1-noise on the right var/value pair, and noise
    # uniformly spread out across others
    emission = generate_language_model(nstate=nvars, nvals=nvals, noise=noise)
    sentences = []
    for i, (z0, zis) in enumerate(zip(vsamples, ssamples)):
        sentence = []
        for zi in zis:
            # zi is the index of the variable in z0
            logits = emission[zi](z0[zi])
            emission_dist = torch.distributions.categorical.Categorical(probs=torch.tensor(logits))
            sentence.append(emission_dist.sample().item())
        sentence = np.array(sentence)
        sentences.append(sentence)
    sentences = np.stack(sentences, axis=0)

    # print the results for verification
    for vsample, ssample, sentence in zip(vsamples, ssamples, sentences):
        print(vsample)
        print(ssample)
        print(sentence)
        print(verbalize(sentence, nvars=nvars, nvals=nvals))

