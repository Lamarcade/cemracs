# this is because my code editor was acting up, so I added this (not necessary)
# isort: skip_file
# fmt: off
import torch
import torch.nn as nn
import torch.nn.functional as F
# fmt: on

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# charge the database of CO2 data and assign condition classes
files_and_classes = [
    ("chart_data-3.csv", 0, "< 2°"),
    ("chart_data-4.csv", 1, "= 1.5°"),
    ("chart_data-2.csv", 2, "> 2°"),
]

all_dfs = []
for filename, cls, _ in files_and_classes:
    all_dfs.append(pd.read_csv(filename))

# we find the common years across all data files
common_cols = set.intersection(*[set(df.columns) for df in all_dfs])
annees = [c for c in common_cols if c.isdigit()]

# we drop the years with missing data (NaNs) to avoid interpolation
df_concat = pd.concat([df[annees] for df in all_dfs])
valid_years = df_concat.dropna(axis=1).columns.tolist()
valid_years.sort(key=int)
annees = valid_years

all_data = []
all_labels = []

for i, (_, cls, _) in enumerate(files_and_classes):
    data_array = all_dfs[i][annees].values
    all_data.append(data_array)
    all_labels.append(np.full(len(data_array), cls))

# we combine all data, do a numpy conversion and apply normalization
data_array = np.vstack(all_data)
labels_array = np.concatenate(all_labels)

mean = data_array.mean(axis=0)
std = data_array.std(axis=0)
data_norm = (data_array - mean) / std

# we convert the normalized data and condition labels into torch tensors
dataset_tensor = torch.tensor(data_norm, dtype=torch.float32)
labels_tensor = torch.tensor(labels_array, dtype=torch.long)
N_DIM = dataset_tensor.shape[1]
N_CLASSES = len(files_and_classes)


# we sample random numbers from a normal distribution to initialize x0
def sample_source(n_samples):
    return torch.randn(n_samples, N_DIM)


# we sample random real data to use as target x1 at t = T, along with their condition
# we do this for a batch of n_samples at once
def sample_target(n_samples):
    indices = torch.randint(0, len(dataset_tensor), (n_samples,))
    return dataset_tensor[indices], labels_tensor[indices]


# the neural network u_t^\theta(x, c) predicting the vector field
# input: N_DIM of data + 1 for the time + N_CLASSES for the one-hot condition
model = nn.Sequential(
    nn.Linear(N_DIM + 1 + N_CLASSES, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, N_DIM),  # the output is the vector field (speed) for each dimension
)

# we choose a learning rate : that works like a "delta t", not too big otherwise it will not be good,
# but not too small otherwise it will be too slow
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# loop of training
batch_size = 32  # a batch size of 32 is sufficient for this dataset size
for step in range(3000):
    x0 = sample_source(batch_size)
    x1, c = sample_target(batch_size)
    t = torch.rand(batch_size, 1)

    # we apply the flow matching equations
    xt = (1 - t) * x0 + t * x1
    u_target = x1 - x0

    # we concatenate the spatial dimensions (N_DIM), time (1), and one-hot condition (N_CLASSES)
    c_onehot = F.one_hot(c, num_classes=N_CLASSES).float()
    txc = torch.cat([xt, t, c_onehot], dim=1)

    u_pred = model(txc)
    loss = torch.mean((u_pred - u_target) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# we solve the ODE to generate "numb_generate" new trajectories for each condition separately
numb_generate = 200
generated_data = {}
with torch.no_grad():
    for c_idx in range(N_CLASSES):
        x = sample_source(numb_generate)
        c = torch.full((numb_generate,), c_idx, dtype=torch.long)
        c_onehot = F.one_hot(c, num_classes=N_CLASSES).float()

        n_steps = 100
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((numb_generate, 1), i * dt)
            txc = torch.cat([x, t, c_onehot], dim=1)
            x = x + model(txc) * dt

        # we reverse the normalization so the data returns to its original scale
        generated_data[c_idx] = x.numpy() * std + mean

# plot the trajectories for each condition
fig, axes = plt.subplots(
    N_CLASSES, 2, figsize=(14, 4 * N_CLASSES), sharey=True, sharex=True
)
x_axis = [int(a) for a in annees]

for c_idx, (_, _, name) in enumerate(files_and_classes):
    ax_real = axes[c_idx, 0]
    ax_gen = axes[c_idx, 1]

    # real data on the left
    reels_c = dataset_tensor[labels_tensor == c_idx].numpy() * std + mean
    for i in range(min(len(reels_c), 200)):
        ax_real.plot(x_axis, reels_c[i], "b-", alpha=0.2)
    ax_real.set_title(f"real data: {name}")

    # generated data on the right
    gen_c = generated_data[c_idx]
    for i in range(len(gen_c)):
        ax_gen.plot(x_axis, gen_c[i], "r-", alpha=0.2)
    ax_gen.set_title(f"flow matching: {name}")

plt.tight_layout()
plt.savefig("result_csv_conditional.png")
