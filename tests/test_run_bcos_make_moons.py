# keep in mind the seed in the sampling
# Add extraction time
# Add annealing time
# For quantum (add annealing time)

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons

from src.models.BQMBuilder import BcosQmatPaper
from src.models.QuboSolver import QuboSolver

percentage_kept = 0.75
cores = -1
batch_size = 80
num_reads = 2000
 
X, Y = make_moons(n_samples=2000, noise=0.2, random_state=42)
 
# TEST WITH SOME PLOT
model = QuboSolver(X, Y)
results = model.run_QuboSolver(BcosQmatPaper, percentage_keep=percentage_kept)
print(sum(results['results'].values())/X.shape[0])
 
df = pd.DataFrame(X, columns=["x", "y"])
df["class"] = Y  # Class 0 or 1
df['kept'] = list(results['results'].values())

df["class_kept"] = df["class"].astype(str) + "_" + df["kept"].astype(str)
df["point_size"] = df["kept"].apply(lambda x: 50 if x == 1 else 100)
# Define color mapping

# 🔹 New Improved Color Palette
palette = {
    "0_1": "blue",    # Kept - Class 0
    "1_1": "orange",  # Kept - Class 1
    "0_0": "green",   # Removed - Class 0
    "1_0": "red"      # Removed - Class 1
}

# Define legend labels
legend_labels = {
    "0_1": "Class 0 - Kept",
    "1_1": "Class 1 - Kept",
    "0_0": "Class 0 - Removed",
    "1_0": "Class 1 - Removed"
}

# Assign size: Larger for removed points
df["point_size"] = df["kept"].apply(lambda x: 50 if x == 1 else 100)

# Create a single plot
fig, ax = plt.subplots(figsize=(8, 8))

# Plot the half-moons dataset
sns.scatterplot(
    ax=ax,
    x="x", y="y",
    hue="class_kept",
    palette=palette,
    size="point_size",  # Assign different sizes
    sizes=(50, 100),    # Explicit size mapping (Kept: 50, Removed: 100)
    data=df
)

# Set title
percentage_kept = df["kept"].mean()  # Compute kept percentage
ax.set_title(f"Half-Moons Data with instance selection reduction = {1 - percentage_kept:.2f}")

# Modify legend (fix KeyError)
handles, labels = ax.get_legend_handles_labels()
filtered_handles_labels = [(h, legend_labels[l]) for h, l in zip(handles, labels) if l in legend_labels]  # Filter valid labels

# Unpack filtered handles & labels
if filtered_handles_labels:  
    filtered_handles, filtered_labels = zip(*filtered_handles_labels)
    ax.legend(filtered_handles, filtered_labels, title="Legend", loc="upper right")

plt.tight_layout()
plt.show()
  