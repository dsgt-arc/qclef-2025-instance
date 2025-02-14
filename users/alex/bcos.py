# keep in mind the seed in the sampling
# Add extraction time
# Add annealing time
# For quantum (add annealing time)

import pandas as pd
import QuboSolver as qs
from BQMBuilder import BcosQmatPaper

 
percentage_kept = 0.9
cores = 12
batch_size = 80
num_reads = 2000

data = pd.read_csv("data/vader_nyt/bert_nytEditorialSnippets.csv")
X = data.iloc[:, 1:-2].values
Y = data.iloc[:, -2].values

model = qs.QuboSolver(X, Y)
results = model.run_QuboSolver(BcosQmatPaper, percentage_keep=percentage_kept)
print(sum(results['results'].values())/X.shape[0])

 


print('a')
 
 
 