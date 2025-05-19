from src import utils
import pandas as pd
import yaml
from src.models.BQMBuilder import BcosQmatPaper, SVC_diagonal, IterativeDeletion, SVC_iterative_combined
from src.models.RandomSolver import RandomSolver
from src.models.QuboSolver import QuboSolver
#from src.models.Evaluator import Evaluator
from box import ConfigBox
from sklearn.datasets import load_svmlight_file
from tqdm import tqdm
import pdb
import numpy as np
 
# Load config
with open("team_workspace/2A/qclef-2025-instance/config/config_bcos_run.yml", "r") as file:
    config = ConfigBox(yaml.safe_load(file))

filepath = 'team_workspace/2A/qclef-2025-instance/outputs/'
dataset = 'Vader'
method = 'SA'
submissionID = 'combined_075' #bcos, random, svc, it_del

if dataset == "Yelp":
    filedir =  'yelp_reviews_2L'

if dataset == "Vader":
    filedir = "vader_nyt_2L"

results_per_fold = []
for i in tqdm(range(5)):
    X, y = load_svmlight_file(f"datasets/Tasks/2A/{filedir}/llama/train{i}.gz", n_features=4096)
 
    X_dense = pd.DataFrame(X.toarray())

    original_indices = np.arange(len(y))
    shuffled_indices = np.random.permutation(len(y))
    X_dense_shuffled = X_dense.iloc[shuffled_indices].reset_index(drop=True)
    y_shuffled = y[shuffled_indices]
    inverse_indices = np.argsort(shuffled_indices)   

    is_model = QuboSolver(X=X_dense_shuffled, Y=y_shuffled, sampler=method, **config.instance_selection)
    is_results = is_model.run_QuboSolver(SVC_iterative_combined)
    selected_samples_shuffled = [key for key, value in is_results['results'].items() if value == 1]
    results_per_fold.append(is_results)
    
    selected_samples_original = [shuffled_indices[sample] for sample in selected_samples_shuffled]

    filename = f"{filepath}{dataset}_{i}_{method}_{'ds-at-gt-qclef'}_{submissionID}.txt"
        
    with open(filename, 'w') as file:
        # Write the keys
        for sample_point in selected_samples_original:
            file.write(f"{sample_point}\n")
         
        file.write(f"{str(is_results['problem_ids'])}\n")

pdb.set_trace()
 