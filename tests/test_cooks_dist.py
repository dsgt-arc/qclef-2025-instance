from src import utils
import pandas as pd
import yaml
from src.models.BQMBuilder import BcosQmatPaper, IterativeDeletion
from src.models.QuboSolver import QuboSolver
from src.models.RandomSolver import RandomSolver
from src.models.Evaluator import Evaluator
from box import ConfigBox

with open("config/config_bcos_run.yml", "r") as file:
    config = ConfigBox(yaml.safe_load(file))

data = pd.read_csv("data/raw_yelp_sample.csv")
X = data.iloc[:, 2:-1].values
Y = data.iloc[:, -1].values

data = utils.k_fold(X, Y, **config.data, printstats=True)

data_embed = pd.read_csv("data/bert_yelp_sample.csv")
X_embed = data_embed.iloc[:, 2:-1].values
Y_embed = data_embed.iloc[:, -1].values

data_embed = utils.k_fold(X_embed, Y_embed, **config.data, printstats=True)

data_is_baseline = []
data_is_cooks = []

for fold in range(len(data)):
    baseline = RandomSolver(data[fold][0], data[fold][1], percentage_keep=config.instance_selection.percentage_keep, random_state=config.instance_selection.random_state)
    sampled_X_baseline, sampled_Y_baseline = baseline.run_solver()  

    cooks_model = QuboSolver(data_embed[fold][0], data_embed[fold][1], **config.instance_selection)
    cooks_results = cooks_model.run_QuboSolver(IterativeDeletion)
    sampled_X_cooks = data[fold][0][cooks_results['sampled_X_idx'], :]
    sampled_Y_cooks = data[fold][1][cooks_results['sampled_Y_idx']]
 
    is_folds_baseline = (sampled_X_baseline, sampled_Y_baseline, data[fold][2], data[fold][3], data[fold][4], data[fold][5])
    is_folds_cooks = (sampled_X_cooks, sampled_Y_cooks, data[fold][2], data[fold][3], data[fold][4], data[fold][5])
    
    data_is_baseline.append(is_folds_baseline)
    data_is_cooks.append(is_folds_cooks)
    
    print(f'<TRAIN DATA>: shape original X data in fold {fold}: {data[0][0].shape}; Random Baseline reduced X data shape: {sampled_X_baseline.shape} ')

eval_cooks =  Evaluator(orig_folds=data, is_folds=data_is_cooks, config=config)
results_cooks = eval_cooks.cross_validation()

eval_baseline = Evaluator(orig_folds = data, is_folds=data_is_baseline, config=config)
results_baseline = eval_baseline.cross_validation()

print(f'Cooks: {results_cooks}')
print(f'Baseline: {results_baseline}')