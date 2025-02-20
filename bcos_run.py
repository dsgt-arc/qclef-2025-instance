from src import utils
import pandas as pd
import yaml
from src.models.BQMBuilder import BcosQmatPaper
from src.models.QuboSolver import QuboSolver
from box import ConfigBox

# Load and split data
data = pd.read_csv("data/vader_nyt/bert_nytEditorialSnippets.csv")
X = data.iloc[:, 1:-2].values
Y = data.iloc[:, -2].values

# data is a list of tuples. Each tuple looks like this: (X_train, y_train, X_val, y_val, X_test, y_test)
# X_val, y_val should go in the transformer's validation set, X_test and y_test should be untouched and used for true out of sample evaluation as in the paper Table 3 & Figure 3
data = utils.k_fold(X, Y, printstats=True)

# Run instance selection
with open("config/config_bcos_run.yml", "r") as file:
    config = ConfigBox(yaml.safe_load(file))

# Example: Bcos instance selection results for the first fold
bcos_model = QuboSolver(data[0][0], data[0][1], **config.instance_selection)
bcos_results = bcos_model.run_QuboSolver(BcosQmatPaper)
print(f"Achieved sample size reduction: {sum(bcos_results['results'].values())/data[0][0].shape[0]}")
print(f"Annealing Time: {bcos_results['annealing_time_total']}")
print(f"BQM Building Time: {bcos_results['building_time']}")


 
