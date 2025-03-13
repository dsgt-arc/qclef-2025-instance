import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.metrics import f1_score
from typing import List, Dict
import random
from torch.nn.utils.rnn import pad_sequence

class BaselineEmbeddingModel(torch.nn.Module):
    """Baseline model to use with pre-computed embeddings"""
    def __init__(self, input_dim, num_labels):
        super(BaselineEmbeddingModel, self).__init__()
        # model layer for classification on embeddings
        self.fc = torch.nn.Linear(input_dim, num_labels)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, embeddings):
        # embeddings shape: (batch_size, embedding_dim)
        x = self.fc(embeddings)
        return self.softmax(x)

class TokenizedDataset(Dataset):
    """PyTorch dataset for pre-tokenized text samples.
    
    Attributes:
        data: List of dictionaries with 'input_ids' (embeddings) and 'label' keys.
    """
    def __init__(self, X, y):
        
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.X[idx], dtype=torch.float32),
            "label": torch.tensor(self.y[idx], dtype=torch.long),
        }

def collate_fn(batch):
    """Collate function to dynamically pad sequences (if needed)."""
    input_ids = [item["input_ids"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    input_ids = torch.stack(input_ids)

    return {"input_ids": input_ids, "label": labels}

class BaselineTrainer:
    """Trainer for the baseline model

    Attributes:
        data: The reduced dataset after performing instacne selection
        """
    def __init__(self, X, y, input_dim: int, num_labels: int, device: str = None):
        self.X = X
        self.y = y
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BaselineEmbeddingModel(input_dim, num_labels).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, batch_size: int = 8, epochs: int = 3, lr: float = 5e-5):
        """
        Runs the training loop for the baseline model
        """

        # Create DataLoader
        dataset = TokenizedDataset(self.X, self.y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        optimizer = AdamW(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            all_preds, all_labels = [], []

            for batch in dataloader:
                embeddings = batch["input_ids"].to(self.device)  # Use precomputed embeddings directly
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()

                # Feed embeddings to the model
                outputs = self.model(embeddings)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

            macro_f1 = f1_score(all_labels, all_preds, average="macro")
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}, Macro F1: {macro_f1:.4f}")
