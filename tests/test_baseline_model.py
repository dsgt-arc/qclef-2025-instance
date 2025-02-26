from src.models.Evaluator import Evaluator
from src.models.RandomSolver import RandomSolver
from src import utils
import numpy as np
import pandas as pd
import torch
import yaml
from box import ConfigBox

# Load config
with open("config/config_bcos_run.yml", "r") as file:
    config = ConfigBox(yaml.safe_load(file))

data = pd.read_csv("data/bert_nytEditorialSnippets.csv")

X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

orig_folds = utils.k_fold(X, y,  **config.data, printstats=True)
is_folds = orig_folds.copy()

#Run instance selection first for each fold
for i in range(len(orig_folds)):
    solver = RandomSolver(orig_folds[i][0], orig_folds[i][1])

    #Replace X_train and y_train with reduced data for is_folds
    fold_list = list(is_folds[i])
    fold_list[0], fold_list[1] = solver.run_solver()
    is_folds[i] = tuple(fold_list)

evaluator = Evaluator(orig_folds, is_folds)
results = evaluator.cross_validation()
print(results)