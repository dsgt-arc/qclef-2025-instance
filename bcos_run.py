from src import utils
import pandas as pd
import yaml
from src.models.BQMBuilder import BcosQmatPaper, IterativeDeletion, SVC_diagonal
from src.models.QuboSolver import QuboSolver
from src.models.RandomSolver import RandomSolver
from src.models.Evaluator import Evaluator
from box import ConfigBox

# Load config
with open("config/config_bcos_run.yml", "r") as file:
    config = ConfigBox(yaml.safe_load(file))

# Need to load and run the bcos method here first, before passing it further
  
# Load and split  text data
# data_raw_text = pd.read_csv("data/vader_nyt/raw_binned_nytEditorialSnippets.csv")
# X = data_raw_text.iloc[:, 3:-1].values
# Y = data_raw_text.iloc[:, -1].values

# data_embeddings = pd.read_csv("data/vader_nyt/bert_nytEditorialSnippets.csv",index_col=0)
# X_embedding = data_embeddings.iloc[:,0:-2].values
# Y_embedding = data_embeddings.iloc[:,-2].values

data_raw_text = pd.read_csv("data/raw_binned_nytEditorialSnippets.csv")
X = data_raw_text.iloc[:, 2:-1].values
Y = data_raw_text.iloc[:, -1].values

data_embeddings = pd.read_csv("data/bert_nytEditorialSnippets.csv",index_col=0)
X_embedding = data_embeddings.iloc[:,2:-1].values
Y_embedding = data_embeddings.iloc[:,-1].values

# data is a list of tuples. Each tuple looks like this: (X_train, y_train, X_val, y_val, X_test, y_test)
# X_val, y_val should go in the transformer's validation set, X_test and y_test should be untouched and used for true out of sample evaluation as in the paper Table 3 & Figure 3
data_raw_text, data_embeddings = utils.k_fold(X_raw_text=X, y_raw_text=Y, X_embeddings=X_embedding, y_embeddings=Y_embedding, **config.data, printstats=True)

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
    


 
 
