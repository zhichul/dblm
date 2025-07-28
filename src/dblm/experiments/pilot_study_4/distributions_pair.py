
import itertools
import random

def sample_indices(length, n_branches=3, indices=None, model_seed=42, sample_seed=None, debug=False, force_choices=None):
    """
    Randomly samples a sequence of length n/2 where each element is a pair from a list of indices,
    without replacement. If the list of indices is not specified, it is taken to be list(range(2*length)).

    The generating process works as follows: at each position, sample uniformly randomly a subset of
    size n_branches from remaining indices, then sample uniformly from this subset to pick the next index.
    The random subset is determined by a model_seed as well as the prefix of indices,
    and can be intepreted as part of the model. (Specifically, one can think about first sampling
    the subset for every possible prefix, then the randomness left in the generating process is picking
    from each subset, but since there's exponentially many possible prefixes, precomputing all of it
    is expensive, so we instead use seeds to procedurally recreate only the necessary subsets.
    """
    if sample_seed is None:
        sample_seed = random.random()
    indices_ = list(range(length * 2)) if indices is None else indices  # store the old indices
    indices =  list(range(len(indices_)))
    local_choices = tuple()
    choices = tuple()
    for i in range(length):
        # sample subset
        random.seed(hash((model_seed, choices)))
        options = [(indices[a], indices[b]) for a in range(len(indices)) for b in range(a+1, len(indices))]
        subset = random.sample(options, k=min(n_branches, len(options)))

        # uniformly or otherly sample from subset
        random.seed(hash((sample_seed, choices)))
        local_choice = random.choice(list(range(len(subset)))) if force_choices is None else force_choices[i]
        choice = subset[local_choice]
        local_choices += (local_choice,)
        choices += (choice,)
        if debug:
            print(f"step={i} subset={subset} choice={choice}")

        # remove from indices
        indices = [idx for idx in indices if idx not in choice]
    return tuple((indices_[c1], indices_[c2]) for (c1, c2) in choices), local_choices


def all_indices(length, n_branches=3, indices=None, model_seed=42, debug=False):
    indices = []
    for force_choices in itertools.product(*([list(range(min(n_branches,(2*i * (2*i-1) // 2)))) for i in reversed(range(1,length+1))])): # type:ignore
        choices, _ = sample_indices(length, n_branches=n_branches, model_seed=model_seed, debug=debug, force_choices=force_choices)
        indices.append(choices)
    return indices

if __name__ == "__main__":
    N = 100
    branch=13
    model_seed=42
    length=5
    random.seed(42)
    sample_seeds = [random.random() for _ in range(N)]
    indices = []
    for i in range(N):
        choices, local_choices = sample_indices(length, n_branches=branch, model_seed=model_seed, sample_seed=sample_seeds[i], debug=False)
        indices.append(choices)

    for index in sorted(indices):
        print(index)
    ai = all_indices(5, n_branches=branch, model_seed=42, debug=False)
    print(len(ai))
