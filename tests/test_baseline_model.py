from src.models.BaselineModel import TokenizedDataset, RandomSamplingTrainer
import numpy as np
import pandas as pd
import torch

def load_csv_in_xy_format(csv_path: str):
    df = pd.read_csv(csv_path, index_col=0)
    
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)

    return X, y

X, y = load_csv_in_xy_format("data/bert_nytEditorialSnippets.csv")

data = [{"input_ids": torch.tensor(X[i].tolist(), dtype=torch.float32), "label": int(y[i])} for i in range(len(y))]

input_dim = X.shape[1]

# Set hyperparameters
sample_ratio = 0.5
batch_size = 4
epochs = 3

# Initialize the RandomSamplingTrainer
trainer = RandomSamplingTrainer(input_dim=input_dim, num_labels=2)
trainer.train(data=data, sample_ratio=sample_ratio, batch_size=batch_size, epochs=epochs)

# Save the trained model
torch.save(trainer.model.state_dict(), "users/chloe/models/nytEditorials_bert_baseline_model")
