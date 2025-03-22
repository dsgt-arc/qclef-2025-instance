from src import utils
import pandas as pd
import yaml
import random

from sklearn.datasets import make_blobs
from src.models.BQMBuilder import BcosQmatPaper
from src.models.QuboSolver import QuboSolver
from src.models.RandomSolver import RandomSolver
from src.models.Evaluator import Evaluator
from box import ConfigBox
import numpy as np

# Load config
with open("config/config_bcos_run_org_example.yml", "r") as file:
    config = ConfigBox(yaml.safe_load(file))
    
 
random_state=0

random.seed(random_state)

num_points = 500
num_classes = 5

# Creating our dataset of 500 2D points belonging to 5 classes
X, y = make_blobs(n_samples=num_points, centers=num_classes, n_features=2,
                  random_state=random_state)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
for i in range(num_classes):
    plt.scatter(X[y == i, 0], X[y == i, 1], label=f'Class {i}', alpha=0.7)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('2D Points with 5 Classes')
plt.legend()
plt.grid(True)
plt.show()


bcos_model = QuboSolver(X, y, **config.instance_selection)
bcos_results = bcos_model.run_QuboSolver(BcosQmatPaper)
 

selected_points = list(bcos_results['results'].values())
colors = ['blue', 'green', 'red', 'orange', 'magenta']

# Plot the selected points
plt.figure(figsize=(10, 8))
for i in range(num_classes):
    # Original points with lower alpha
    plt.scatter(X[y == i, 0], X[y == i, 1], label=f'Class {i}', alpha=0.3, color=colors[i])

    # Selected points with higher alpha and the same color
    plt.scatter(X[(y == i) & (np.isin(np.arange(num_points), selected_points)), 0],
                X[(y == i) & (np.isin(np.arange(num_points), selected_points)), 1],
                alpha=1, color=colors[i],marker='x')  # Use the same color

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Selected Points using Simulated Annealing')
plt.legend()
plt.grid(True)
plt.show()


print('a')