import torch

import tqdm
from dblm.utils import seeding
import modeling
import data
from dblm.core.modeling import factor_graphs

# Sample estimated nested loss: True nested model: average loss is 4.8037500429153415 with 100 samples.

# just in theory the true interleaved model (joint not ) has
# 1.184996437312091 * 0.8 entropy in z0
# 2.0794415416798357 * 0.2 entropy in noise z0
# 4.003219388305503 entropy in mixture (5 + 3) switch zs with 0.8 0.2 split
# 5.493061443340547 * 0.2 entropy in the noise z>0
# some positive entropy * 0.8 in the z>0
def main():
    seed = 42
    seeding.seed(seed)
    torch.set_anomaly_enabled(True)
    true_nested_model, observables = modeling.generate_model()  # type:ignore
    true_nested_model = factor_graphs.BPAutoregressiveIncompleteLikelihoodFactorGraph.from_factor_graph(true_nested_model, observables) # type:ignore
    true_interleaved_model, _ = modeling.generate_model(directed=True)
    observable_vars = [var for _, var in observables]

    # evaluation setup
    sample_size = 100
    indices = list(range(sample_size))
    dataset = data.generate_data(sample_size, true_interleaved_model, observable_vars, seed=seed)

    # evaluate
    bar = tqdm.tqdm(indices)
    epoch_loss = 0
    for i in bar:
        loss = -true_nested_model.log_marginal_probability(assignment=list(zip(observable_vars, dataset[i].tolist())))
        # bookkeep
        epoch_loss += loss.item() / len(indices)
        bar.set_description_str(f"loss={loss.item():.2}")
    print(f"True nested model: average loss is {epoch_loss} with {sample_size} samples.")

if __name__ == "__main__":
    main()
