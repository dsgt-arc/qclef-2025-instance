from src import utils
import pandas as pd
import yaml
from src.models.BQMBuilder import BcosQmatPaper, IterativeDeletion, SVC_diagonal
from src.models.QuboSolver import QuboSolver
from src.models.RandomSolver import RandomSolver
from src.models.Evaluator import Evaluator
from box import ConfigBox
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import gzip
import numpy as np

# Load config
with open("config/config_bcos_run.yml", "r") as file:
    config = ConfigBox(yaml.safe_load(file))

#Organizer's data
data_embeddings = []

for i in range(5):
    with gzip.open(f'data/nyt/train{i}.gz', 'rb') as f:
        X_sparse, y = load_svmlight_file(f)

    X_dense = X_sparse.toarray()

    X_train, X_test, y_train, y_test = train_test_split(X_dense, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    data_embeddings.append((X_train, y_train, X_val, y_val, X_test, y_test))

data_raw_text = []

for i in range(5):
    df = pd.read_parquet(f'data/nyt/train_fold_{i}.parquet')

    X_train, X_test, y_train, y_test = train_test_split(np.array(df['text']), np.array(df['label']), test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    data_raw_text.append((X_train, y_train, X_val, y_val, X_test, y_test))

data_is_baseline_raw = []
data_is_baseline_embed = []
data_is_bcos_raw = []
data_is_bcos_embed = []
data_is_iterative_deletion_raw = []
data_is_iterative_deletion_embed = []
data_is_SVC_diagonal_model_results_raw = []
data_is_SVC_diagonal_model_results_embed = []

method = 'SA-local' #SA, QA

# Prepare IS data folds
for fold in range(len(data_raw_text)):
#for fold in range(1):
    # For the random solver, it does not matter if we sample randomly on the 
    baseline_raw = RandomSolver(data_raw_text[fold][0], data_raw_text[fold][1], percentage_keep=config.instance_selection.percentage_keep, random_state=config.instance_selection.random_state)
    sampled_X_baseline_raw, sampled_Y_baseline_raw = baseline_raw.run_solver()
    
    baseline_embed = RandomSolver(data_embeddings[fold][0], data_embeddings[fold][1], percentage_keep=config.instance_selection.percentage_keep, random_state=config.instance_selection.random_state)
    sampled_X_baseline_embed, sampled_Y_baseline_embed = baseline_embed.run_solver()    
 
    bcos_model = QuboSolver(data_embeddings[fold][0], data_embeddings[fold][1], sampler=method, **config.instance_selection)
    bcos_results = bcos_model.run_QuboSolver(BcosQmatPaper)
    
    iterative_deletion_model = QuboSolver(data_embeddings[fold][0], data_embeddings[fold][1], sampler=method, **config.instance_selection)
    iterative_deletion_results = iterative_deletion_model.run_QuboSolver(IterativeDeletion)
    
    SVC_diagonal_model =  QuboSolver(data_embeddings[fold][0], data_embeddings[fold][1], sampler=method, **config.instance_selection) #add all the relevant things to the config.
    SVC_diagonal_model_results = SVC_diagonal_model.run_QuboSolver(SVC_diagonal) 
    
    # --- need to take the appropriate indices here and now
    sampled_indices_X_bcos = bcos_results['indices_X']
    sampled_indices_y_bcos = bcos_results['indices_y']
   
    sampled_indices_X_iterative_deletion = iterative_deletion_results['indices_X']
    sampled_indices_y_iterative_deletion= iterative_deletion_results['indices_y']
     
    sampled_indices_X_SVC = SVC_diagonal_model_results['indices_X']
    sampled_indices_y_SVC = SVC_diagonal_model_results['indices_y']
     
    is_folds_baseline_raw = (sampled_X_baseline_raw, sampled_Y_baseline_raw, data_raw_text[fold][2], data_raw_text[fold][3], data_raw_text[fold][4], data_raw_text[fold][5])
    is_folds_baseline_embed = (sampled_X_baseline_embed, sampled_Y_baseline_embed, data_embeddings[fold][2], data_embeddings[fold][3], data_embeddings[fold][4], data_embeddings[fold][5])
    is_folds_bcos_raw = (data_raw_text[fold][0][sampled_indices_X_bcos], data_raw_text[fold][1][sampled_indices_y_bcos], data_raw_text[fold][2], data_raw_text[fold][3], data_raw_text[fold][4], data_raw_text[fold][5])
    is_folds_bcos_embed = (data_embeddings[fold][0][sampled_indices_X_bcos], data_embeddings[fold][1][sampled_indices_y_bcos], data_embeddings[fold][2], data_embeddings[fold][3], data_embeddings[fold][4], data_embeddings[fold][5])
    is_folds_iterative_deletion_raw = (data_raw_text[fold][0][sampled_indices_X_iterative_deletion], data_raw_text[fold][1][sampled_indices_y_iterative_deletion], data_raw_text[fold][2], data_raw_text[fold][3], data_raw_text[fold][4], data_raw_text[fold][5])
    is_folds_iterative_deletion_embed = (data_embeddings[fold][0][sampled_indices_X_iterative_deletion], data_embeddings[fold][1][sampled_indices_y_iterative_deletion], data_embeddings[fold][2], data_embeddings[fold][3], data_embeddings[fold][4], data_embeddings[fold][5])
    is_folds_svc_distance_raw = (data_raw_text[fold][0][sampled_indices_X_SVC], data_raw_text[fold][1][sampled_indices_y_SVC], data_raw_text[fold][2], data_raw_text[fold][3], data_raw_text[fold][4], data_raw_text[fold][5])
    is_folds_svc_distance_embed = (data_embeddings[fold][0][sampled_indices_X_SVC], data_embeddings[fold][1][sampled_indices_y_SVC], data_embeddings[fold][2], data_embeddings[fold][3], data_embeddings[fold][4], data_embeddings[fold][5])
    
    data_is_baseline_raw.append(is_folds_baseline_raw)
    data_is_baseline_embed.append(is_folds_baseline_embed)
    data_is_bcos_raw.append(is_folds_bcos_raw)
    data_is_bcos_embed.append(is_folds_bcos_embed)
    data_is_iterative_deletion_raw.append(is_folds_iterative_deletion_raw)
    data_is_iterative_deletion_embed.append(is_folds_iterative_deletion_embed)
    data_is_SVC_diagonal_model_results_raw.append(is_folds_svc_distance_raw)
    data_is_SVC_diagonal_model_results_embed.append(is_folds_svc_distance_embed)
    
    print(f'<TRAIN DATA>: shape original X data in fold {fold}: {data_raw_text[fold][0].shape}; Random Baseline reduced X data shape: {sampled_X_baseline_raw.shape} ')
    
eval_full =  Evaluator(orig_folds=data_raw_text, is_folds_raw=data_raw_text, is_folds_embed=data_embeddings, config=config, model_type="logreg")
results_full = eval_full.cross_validation()

eval_baseline = Evaluator(orig_folds = data_raw_text, is_folds_raw=data_is_baseline_raw, is_folds_embed=data_is_baseline_embed, config=config, model_type="logreg")
results_baseline = eval_baseline.cross_validation()

eval_bcos = Evaluator(orig_folds = data_raw_text, is_folds_raw=data_is_bcos_raw, is_folds_embed=data_is_bcos_embed, config=config, model_type="logreg")
results_bcos = eval_bcos.cross_validation()

eval_cooksD = Evaluator(orig_folds = data_raw_text, is_folds_raw=data_is_iterative_deletion_raw, is_folds_embed=data_is_iterative_deletion_embed, config=config, model_type="logreg")
results_cooksD = eval_cooksD.cross_validation()

eval_svc = Evaluator(orig_folds = data_raw_text, is_folds_raw=data_is_SVC_diagonal_model_results_raw, is_folds_embed=data_is_SVC_diagonal_model_results_embed, config=config, model_type="logreg")
results_svc = eval_svc.cross_validation()

print(f'Full sample: {results_full}')
print(f'Baseline: {results_baseline}')
print(f'Bcos: {results_bcos}')
print(f'CooksD: {results_cooksD}')
print(f'SVC: {results_svc}')
print('a')
    


 
 
