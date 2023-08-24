
import random
import torch
import tqdm
from dblm.core.modeling import factor_graphs
from dblm.utils import seeding
import modeling
import data



def main():
    seed = 42
    seeding.seed(seed)
    torch.set_anomaly_enabled(True)
    (true_interleaved_model, observables) = modeling.generate_model(directed=True)  # type:ignore
    (random_interleaved_model, _) = modeling.generate_model(random=True) # type:ignore
    random_nested_model = factor_graphs.BPAutoregressiveIncompleteLikelihoodFactorGraph.from_factor_graph(random_interleaved_model, observables) # type:ignore
    observable_vars = [var for _, var in observables]

    # training setup
    sample_size = 100
    nepochs = 100
    indices = list(range(sample_size))
    lr = 0.01
    optimizer = torch.optim.Adam(random_nested_model.parameters(), lr)
    dataset = data.generate_data(sample_size, true_interleaved_model, observable_vars, seed=seed)

    print(random_nested_model)

    # train
    bar = tqdm.tqdm(indices)
    for epoch in range(nepochs):
        random.shuffle(indices)
        epoch_loss = 0
        for i in bar:
            loss = -random_nested_model.incomplete_log_likelihood_function(assignment=list(zip(observable_vars, dataset[i].tolist())))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # bookkeep
            epoch_loss += loss.item() / len(indices)
            bar.set_description_str(f"loss={loss.item():.2}")
        print(f"End of epoch {epoch}: average loss is {epoch_loss}")
        if epoch % 10 == 0:
            torch.save(random_nested_model.state_dict(), f"saved_models/nested_model_at_end_of_epoch_{epoch}.pt")

if __name__ == "__main__":
    main()
