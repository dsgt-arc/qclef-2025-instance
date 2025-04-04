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

# Need to load and run the bcos method here first, before passing it further
  
# Load and split  text data
data_raw_text = pd.read_csv("data/vader_nyt/raw_binned_nytEditorialSnippets.csv")
X = data_raw_text.iloc[:, 3:-1].values
Y = data_raw_text.iloc[:, -1].values

data_embeddings = pd.read_csv("data/vader_nyt/bert_nytEditorialSnippets.csv",index_col=0)
X_embedding = data_embeddings.iloc[:,0:-2].values
Y_embedding = data_embeddings.iloc[:,-2].values

# data is a list of tuples. Each tuple looks like this: (X_train, y_train, X_val, y_val, X_test, y_test)
# X_val, y_val should go in the transformer's validation set, X_test and y_test should be untouched and used for true out of sample evaluation as in the paper Table 3 & Figure 3
data_raw_text, data_embeddings = utils.k_fold(X_raw_text=X, y_raw_text=Y, X_embeddings=X_embedding, y_embeddings=Y_embedding, **config.data, printstats=True)

data_is_baseline = []
data_is_bcos = []

method = 'SA-local' #SA, QA

# Prepare IS data folds
#for fold in range(len(data_raw_text)):
for fold in range(1):
    # For the random solver, it does not matter if we sample randomly on the 
    baseline = RandomSolver(data_raw_text[fold][0], data_raw_text[fold][1], percentage_keep=config.instance_selection.percentage_keep, random_state=config.instance_selection.random_state)
    sampled_X_baseline, sampled_Y_baseline = baseline.run_solver()  
 
    bcos_model = QuboSolver(data_embeddings[fold][0], data_embeddings[fold][1], sampler=method, **config.instance_selection)
    bcos_results = bcos_model.run_QuboSolver(BcosQmatPaper)
    
    # --- need to take the appropriate indices here and now
    
    sampled_indices_X_bcos = bcos_results['indices_X']
    sampled_indices_y_bcos = bcos_results['indices_y']
     
    is_folds_baseline = (sampled_X_baseline, sampled_Y_baseline, data_raw_text[fold][2], data_raw_text[fold][3], data_raw_text[fold][4], data_raw_text[fold][5])
    is_folds_bcos = (data_raw_text[fold][0][sampled_indices_X_bcos], data_raw_text[fold][1][sampled_indices_y_bcos], data_raw_text[fold][2], data_raw_text[fold][3], data_raw_text[fold][4], data_raw_text[fold][5])
    
    data_is_baseline.append(is_folds_baseline)
    data_is_bcos.append(is_folds_bcos)
    
    print(f'<TRAIN DATA>: shape original X data in fold {fold}: {data_raw_text[fold][0].shape}; Random Baseline reduced X data shape: {sampled_X_baseline.shape} ')
    

eval_full =  Evaluator(orig_folds=data_raw_text, is_folds=data_raw_text, config=config)
results_full = eval_full.cross_validation()

eval_baseline = Evaluator(orig_folds = data_raw_text, is_folds=data_is_baseline, config=config)
results_baseline = eval_baseline.cross_validation()

eval_bcos = Evaluator(orig_folds = data_raw_text, is_folds=data_is_bcos, config=config)
results_bcos = eval_bcos.cross_validation()

# print(f'Full sample: {results_full}')
print(f'Baseline: {results_baseline}')
print(f'Bcos: {results_bcos}')

print('a')
    


 
 
