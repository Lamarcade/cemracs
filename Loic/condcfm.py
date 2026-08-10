import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

base_dir = Path(__file__).resolve().parent

# charge the database of CO2 data (our own SCE scenarios) and assign condition classes,
# in increasing temperature order
files_and_classes = [
    ("1dot5", 0, "1.5°"),
    ("2", 1, "2°"),
    ("above2", 2, "> 2°"),
]

all_dfs = []
for tag, cls, _ in files_and_classes:
    path = base_dir / "SCE" / f"co2_world_{tag}.csv"
    all_dfs.append(pd.read_csv(path))

# we keep only the series sampled every 5 years (drops the extra annual columns
# that only "above2" has), and find the years common to all data files
common_cols = set.intersection(*[set(df.columns) for df in all_dfs])
annees = sorted(
    (c for c in common_cols if c.isdigit() and int(c) % 5 == 0), key=int
)

# we drop incomplete trajectories (NaNs) row by row instead of interpolating or
# dropping whole years, so the 3 classes keep the same, complete year axis
all_dfs = [df[annees].dropna(axis=0, how="any") for df in all_dfs]

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
for step in tqdm(range(3000), desc="Optimisation", unit="step"):
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
    for c_idx in tqdm(range(N_CLASSES), desc="Simulation", unit="class"):
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

# plot all generated (simulated) trajectories on a single graph, colored by
# temperature scenario: blue = 1.5°C, yellow = 2°C, red = > 2°C
colors = {0: "#1f77b4", 1: "#f2c14e", 2: "#d62728"}

fig, ax = plt.subplots(figsize=(10, 6))
x_axis = [int(a) for a in annees]

for c_idx, (_, _, name) in enumerate(files_and_classes):
    gen_c = generated_data[c_idx]
    for i in range(len(gen_c)):
        ax.plot(x_axis, gen_c[i], color=colors[c_idx], alpha=0.2, linewidth=1)

# one legend entry per class (proxy handles), not one per line
handles = [
    plt.Line2D([0], [0], color=colors[c_idx], lw=2, label=name)
    for c_idx, (_, _, name) in enumerate(files_and_classes)
]
ax.legend(handles=handles, title="Scénario")
ax.set_title("Flow matching : trajectoires générées par scénario de température")
ax.set_xlabel("Année")
ax.set_ylabel("Émissions de CO2 (Mt CO2/yr)")

plt.tight_layout()
output_path = base_dir / "Figs" / "simulations_conditional.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path)