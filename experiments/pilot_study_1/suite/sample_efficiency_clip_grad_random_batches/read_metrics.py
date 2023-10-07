import json
import matplotlib.pyplot as plt

epochs = 27
seeds = [11]#, 12]
for seed in seeds:
    train_loss = []
    dev_loss = []
    for epoch in range(epochs):
        with open(f"saved_models/{seed}/metrics_at_end_of_epoch_{epoch}.json") as f:
            result = json.load(f)
            train_loss.append(result["train_loss"])
            dev_loss.append(result["dev_loss"])
    plt.plot(train_loss, label=f"train_loss_{seed}")
plt.legend()
plt.show()

for seed in seeds:
    dev_loss = []
    for epoch in range(epochs):
        with open(f"saved_models/{seed}/metrics_at_end_of_epoch_{epoch}.json") as f:
            result = json.load(f)
            dev_loss.append(result["dev_loss"])
    plt.plot(dev_loss, label=f"dev_loss_{seed}")
plt.legend()
plt.show()
