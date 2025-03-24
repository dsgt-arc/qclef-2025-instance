from src import utils
import pandas as pd
import yaml
from src.models.BQMBuilder import BcosQmatPaper
from src.models.QuboSolver import QuboSolver
#from src.models.RandomSolver import RandomSolver
#from src.models.Evaluator import Evaluator
from box import ConfigBox
from sklearn.datasets import load_svmlight_file
from tqdm import tqdm
import pdb
 
# Load config
with open("team_workspace/2A/qclef-2025-instance/config/config_bcos_run.yml", "r") as file:
    config = ConfigBox(yaml.safe_load(file))

filepath = 'team_workspace/2A/qclef-2025-instance/outputs/'
dataset = 'vader'
method = 'QA'
submissionID = 'bcos'

results_per_fold = []
for i in tqdm(range(5)):
    X, y = load_svmlight_file(f"datasets/Tasks/2A/vader_nyt_2L/llama/train{i}.gz", n_features=4096)
    X_dense = pd.DataFrame(X.toarray())


    bcos_model = QuboSolver(X=X_dense, Y=y, sampler=method, **config.instance_selection)
    bcos_results = bcos_model.run_QuboSolver(BcosQmatPaper)
    results_per_fold.append(bcos_results)
    
    selected_samples = [key for key, value in bcos_results['results'].items() if value == 1]

    filename = f"{filepath}{dataset}_{i}_{method}_{'ds-at-gt-qclef'}_{submissionID}.txt"
        
    with open(filename, 'w') as file:
        # Write the keys
        for sample_point in selected_samples:
            file.write(f"{sample_point}\n")
         
        file.write(f"{str(bcos_results['problem_ids'])}\n")
 
 