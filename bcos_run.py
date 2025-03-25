from src import utils
import pandas as pd
import yaml
from src.models.BQMBuilder import BcosQmatPaper
from src.models.QuboSolver import QuboSolver
from src.models.RandomSolver import RandomSolver
from src.models.Evaluator import Evaluator
from box import ConfigBox

# Load config
with open("config/config_bcos_run.yml", "r") as file:
    config = ConfigBox(yaml.safe_load(file))

# Load and split data
data = pd.read_csv("data/raw_binned_nytEditorialSnippets.csv")
X = data.iloc[:, 3:-1].values
Y = data.iloc[:, -1].values

# data is a list of tuples. Each tuple looks like this: (X_train, y_train, X_val, y_val, X_test, y_test)
# X_val, y_val should go in the transformer's validation set, X_test and y_test should be untouched and used for true out of sample evaluation as in the paper Table 3 & Figure 3
data = utils.k_fold(X, Y,  **config.data, printstats=True)

data_is_baseline = []
data_is_bcos = []

# Prepare IS data folds
for fold in range(len(data)):
    baseline = RandomSolver(data[fold][0], data[fold][1], percentage_keep=config.instance_selection.percentage_keep, random_state=config.instance_selection.random_state)
    sampled_X_baseline, sampled_Y_baseline = baseline.run_solver()  

    # bcos_model = QuboSolver(data[fold][0], data[fold][1], **config.instance_selection)
    # bcos_results = bcos_model.run_QuboSolver(BcosQmatPaper)
    # sampled_X_bcos = bcos_results['sampled_X']
    # sampled_Y_bcos = bcos_results['sampled_Y']
 
    is_folds_baseline = (sampled_X_baseline, sampled_Y_baseline, data[fold][2], data[fold][3], data[fold][4], data[fold][5])
    # is_folds_bcos = (sampled_X_bcos, sampled_Y_bcos, data[fold][2], data[fold][3], data[fold][4], data[fold][5])
    
    data_is_baseline.append(is_folds_baseline)
    # data_is_bcos.append(is_folds_bcos)
    
    print(f'<TRAIN DATA>: shape original X data in fold {fold}: {data[0][0].shape}; Random Baseline reduced X data shape: {sampled_X_baseline.shape} ')
    

eval_full =  Evaluator(orig_folds=data, is_folds=data, config=config)
results_full = eval_full.cross_validation()

eval_baseline = Evaluator(orig_folds = data, is_folds=data_is_baseline, config=config)
results_baseline = eval_baseline.cross_validation()

# eval_bcos = Evaluator(orig_folds = data, is_folds=data_is_bcos, config=config)
# results_bcos = eval_bcos.cross_validation()

print(f'Full sample: {results_full}')
print(f'Baseline: {results_baseline}')
# print(f'Bcos: {results_bcos}')

print('a')
    


 
 
