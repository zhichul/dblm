# Database Language Models

Codebase for langauge model architectures that condition on externally provided probability distributions, as well as joint architectures that have a graphical model component and a neural lm component.

# Structure

Core interfaces and model definitions are in `src/dblm/core`. Attention mechanisms for encoding distributions is in `src/dblm/rva`. Other folders contain utilities in general or for specific experiments. Experiment related configs and scripts are under `experiments`. 
