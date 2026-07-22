import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

# load CO2 database
files = {0:"all",
        1: "1dot5", 
        2: "2", 
        3: "above2"}

selected = 0

base_dir = Path(__file__).resolve().parent
path = base_dir / "SCE" / f"co2_world_{files[selected]}.csv"

df = pd.read_csv(path)

# Keep only the series sampled every 5 years
annees = [c for c in df.columns if c.isdigit() and int(c) % 5 == 0]

# Drop incomplete trajectories instead of interpolating them
df_annees = df[annees].dropna(axis=0, how="any")

data_array = df_annees.to_numpy(dtype=float)
mean = np.nanmean(data_array, axis=0)
std = np.nanstd(data_array, axis=0)
data_norm = (data_array - mean) / std

dataset_tensor = torch.tensor(data_norm, dtype=torch.float32)
N_DIM = dataset_tensor.shape[1]


# we take N_DIM random numbers to initialize
def sample_source(n_samples):
    return torch.randn(n_samples, N_DIM)


# we take a random real data that we want to put for t = T
# we do that for n_samples pack at once
def sample_target(n_samples):
    indices = torch.randint(0, len(dataset_tensor), (n_samples,))
    return dataset_tensor[indices]


# the neural network or the u_t^\phi(x)
# N_DIM of data + 1 for the time
model = nn.Sequential(
    nn.Linear(N_DIM + 1, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, N_DIM),  # the output is the speed for each dimension (N_DIM)
)

# we choose a learning rate : that work like a "delta t", not too big otherwise it will not be good, but not too small
# otherwise it will be too slow
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# loop of training
batch_size = 32  # we don't have so much data so 32 is suffisent
for step in tqdm(range(3000), desc="Optimisation", unit="step"):
    x0 = sample_source(batch_size)
    x1 = sample_target(batch_size)
    t = torch.rand(batch_size, 1)

    # we apply the flow matching there

    xt = (1 - t) * x0 + t * x1
    u_target = x1 - x0

    # we concatenate the N_DIM of space and the 1 dimension of time
    tx = torch.cat([xt, t.expand(-1, 1)], dim=1)

    u_pred = model(tx)
    loss = torch.mean((u_pred - u_target) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# we resolve the ode for generate "numb_generate" news trajectories
numb_generate = 500
with torch.no_grad():
    x = sample_source(numb_generate)
    n_steps = 100
    dt = 1.0 / n_steps
    for i in tqdm(range(n_steps), desc="Simulation", unit="step"):
        t = torch.full((numb_generate, 1), i * dt)
        tx = torch.cat([x, t.expand(-1, 1)], dim=1)
        x = x + model(tx) * dt

# so that the data returns to its original size
x_generes = x.numpy() * std + mean
x_reels = dataset_tensor.numpy() * std + mean

# plot of trajectories
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
x_axis = [int(a) for a in annees]

# real data on the left
for i in range(len(x_reels)):
    ax1.plot(x_axis, x_reels[i], "b-", alpha=0.2)
ax1.set_title("Real data")

# generate data on the right
for i in range(len(x_generes)):
    ax2.plot(x_axis, x_generes[i], "r-", alpha=0.2)
ax2.set_title("Flow matching")

plt.tight_layout()
output_path = base_dir / "Figs" / f"simulations_{files[selected]}.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path)