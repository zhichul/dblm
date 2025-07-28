from collections import defaultdict
from dblm.experiments.filter_ptb import filter_fn

counter = defaultdict(float)
len_hist = defaultdict(int)
with open("ptb.train.txt") as f:
    for line in f:
        for token in filter(filter_fn, line.strip().split(" ")):
            counter[token] += 1
            len_hist[len(token)] += 1
total = sum(counter.values())
for k in counter:
    counter[k] /= total

with open("vocab_sorted.tsv", "w") as f:
    for k, v in sorted(list(counter.items()), key=lambda x: x[1], reverse=True):
        print(f"{k}\t{v}", file=f)

import matplotlib.pyplot as plt
plt.bar(list(range(max(len_hist.keys()))), [len_hist[k] for k in range(max(len_hist.keys()))])
plt.savefig("lengths.png")
charvocab = set()
for word in counter:
    for char in word:
        charvocab.add(char)
with open("char_sorted.tsv", "w") as f:
    for c in sorted(charvocab):
        print(f"{c}", file=f)