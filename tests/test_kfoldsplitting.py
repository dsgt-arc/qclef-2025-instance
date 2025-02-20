from src import utils
import pandas as pd

data = pd.read_csv("data/vader_nyt/bert_nytEditorialSnippets.csv")
X = data.iloc[:, 1:-2].values
Y = data.iloc[:, -2].values

results = utils.k_fold(X, Y, printstats=True)


 
