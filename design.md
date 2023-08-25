# Summary
The code in this repository implements interleaved autoregressive models and nested models and associated sampling and inference algorithms.

# Design Goals
The primary goals are as follows:
* encourage code reuse
* make changing/adding implementations easier
  
The secondary goals are as follows:
* make the code run fast

# Interfaces
## Distributions
```dblm.core.interfaces.distribution``` contains interfaces capturing the general behavior of the pmf / pdf of (conditional) distributions.

```Distribution``` captures the idea that distributions are functions. The interface subsumes both discrete and continuous distributions. ```fix_variables()``` takes ```dict[int, int]``` of assignments and returns a ```GloballyNormalizedDistribution```.

```LocallyNormalizedDistribution``` captures distributions whose density or pmf is expected to be efficiently computed. It also needs to specify the parent indices and child indices of the distribution.

```GloballyNormalizedDistribution``` captures distributions whose unnormalized density or pmf is expected to be efficiently computed.


## PGMs

```dblm.core.interfaces.pgm``` contains several interfaces for the *pmf*'s of *discrete* distributions representable as PGMs. 

```MultiVariateFunction``` captures the fact that the pmfs of discrete distributions are functions of many variables. It requires that any implementation of such a distribution expose the number of variables that it has and the number of discrete values that each variable can take on. 

Every graphical model representing a discrete distribution over $n$ variables implicitly defines an order over them for indexing.

```ProbabilisticGraphicalModel``` captures some shared behavior to all pgms, namely they:
* ```.graph()``` must have a graphical stucture
* ```.to_factor_graph_model()``` must be convertable to a ```FactorGraphModel```
* ```.to_probability_table()``` can be converted to a (joint) normalized ```ProbabilityTable```
* ```.to_potential_table()``` can be converted to a (joint) unnormalized ```PotentialTable```


```FactorGraphModel``` is the most general representation of a graphical model. This is what we run belief propogation on. It requires the following methods:
* ```factor_variables()``` returns a list of tuples of integer indices identifying the variables associated with the factors.
* ```factor_functions()``` returns a list of ```PotentialTable```'s implementing the factor functions.
* ```conditional_factor_variables(observation)``` returns a list of factor variables that have the variables in the observation removed.
* ```conditional_factor_functions(observations)``` returns a list of factor functions (i.e. ```PotentialTables```) that are conditioned on the observations. For example, suppose the factor graph had a factor over variables 1,2,3 and the observation includes v3=5, then the returned list of factors will have a new ```PotentialTable``` over variables 1,2 that is equivalent to the subtable associated with v3=5, of the original ```PotentialTable```.

```conditional_factor_variables``` and ```conditional_factor_functions``` serves `FactorGraphBeliefPropagation` and serves as utility functions for creating conditional versions of `FactorGraphModel`'s.

```BayesianNetwork``` are directed graphical models that support easy ancestral sampling (such as by using ```AncestralSamplerWithPotentialTables```). Different from a factor graph, directed models are defined with locally normalized conditional probabilitie distributions. It offers the following methods:
* ```local_distributions()``` returns a list of distributions, covering some of the variables of the distribution. (We don't require the list of local distributions to cover all variables of the bayes net. This allows us to define classes that implement pieces of a bayesian network and combine them at a later stage flexibly with each other into complete bayesian networks.)
* ```topological_order()``` returns an ordering (list of indices) of the local distributions such that each local distribution only depends on variables whose definitional probability distribution is lower in the order.
* ```local_variables()``` returns a list of tuples of integer indices identifying the variables associated with the local distributions. Note that these indices include the parent nodes as well as the children nodes.
* ```local_children()``` returns a list of tuples of integer indices identifying the children variables associated with the local distributions. These are the variables whose (conditional) distribution is defined by the local distributions.
* ```local_parents()``` returns a list of tuples of integer indices identifying the parent variables associated with the local distributions. These are the variables whose value are used to choose among different distributions for the children.
* ```parent_indices()```
 returns a tuple of indices indicating the parents of this overall model, defaults to empty tuple (meaning default is the joint distribution).
* ```child_indices()```
 returns a tuple of indices indicating the children of this overall model, defaults to empty tuple (meaning default is the joint distribution).

```MarkovRandomField``` are undirected graphical models that are fairly similar to factor graphs, except they are usually represented differently graphically. The offer the following methods:
* ```local_potentials()``` returns a list of ```PotentialTable```'s implementing the factor functions.
* ```local_variables()``` returns a list of tuples of integer indices identifying the variables associated with the factors.

```PotentialTable``` essentially capture the behavior of nonnegative multivariate functions used in factor graphs as potential functions. The most common way of using it is to use ```log_potential_table()``` which returns a tensor with $n$ dimensions, where the $k$th dimension has size $m_k$ where $m_k$ is the total number of values the $k$th variable can take on. Some implementations of this interface may choose not to represent the materialized table but rather only implement methods like ```log_potential_value(assignment)``` which returns the value corresponding to a specific entry in the table. It has three special methods
* ```condition_on(observation)``` returns the subtable corresponding to clamping the observations.
* ```marginalize_over(variables)``` returns a table after summing out the variables provided in the argument. This is used frequently in belief propagation.
* ```renormalize()``` returns a (jointly) renormalized version of the table, i.e. a ```ProbabilityTable```, whose probabilities are proportional to the unnormalized probabilities in the ```PotentialTable```.

```ProabilityTable``` are noramlized ```PotentialTable```, plus a feature where you can declare a subset of the variables as parent and the remaining children, so the normalization is not always over all the variables, but only over the children variables.
# Implementations