"""
This code visualizes the distribution of H(P) where P is
a random distribution sampled from Dir([1,...,1]).

It stratifies the data by the size of P (number of variables).
"""

import math
import numpy as np
import matplotlib.pyplot as plt
for d, c, s in zip([2,4,8,16,32], ["blue", "green",  "yellow", "brown", "red"], [1,1,1,1,1]):
    ps = np.random.dirichlet(np.array([s] * d),size=100000)
    ents = (-ps * np.log(ps)).sum(axis=1)
    density, edges = np.histogram(ents, density=True, bins=80)
    mids = [(e1+e2)/2 for e1, e2 in zip(edges, edges[1:])]
    plt.plot(mids, density, c=c, label=str(d))
plt.legend()
plt.xticks(np.array([math.log(2) * i for i in range(1,6)]))
plt.show()
