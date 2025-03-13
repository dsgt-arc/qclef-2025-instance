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
        #self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, embeddings):
        # embeddings shape: (batch_size, embedding_dim)
        x = self.fc(embeddings)
        return x #self.softmax(x)

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
    def __init__(self, X_train, y_train, X_val, y_val, input_dim: int, num_labels: int, device: str = None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BaselineEmbeddingModel(input_dim, num_labels).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, batch_size: int = 8, epochs: int = 3, lr: float = 5e-5, patience=5):
        """
        Runs the training loop for the baseline model
        """

        # Create DataLoader
        dataset_train = TokenizedDataset(self.X_train, self.y_train)
        dataset_val = TokenizedDataset(self.X_val, self.y_val)
        
        train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        optimizer = AdamW(self.model.parameters(), lr=lr)

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_model_state = None
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            all_preds, all_labels = [], []

            for batch in train_dataloader:
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

                #print(list(self.model.parameters())[0])
    
            train_f1 = f1_score(all_labels, all_preds, average="macro")
            avg_train_loss = total_loss / len(train_dataloader)

            # Validation Step
            self.model.eval()
            val_loss = 0
            val_preds, val_labels = [], []

            with torch.no_grad():
                for batch in val_dataloader:
                    embeddings = batch["input_ids"].to(self.device)
                    labels = batch["label"].to(self.device)

                    outputs = self.model(embeddings)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_labels.extend(labels.cpu().numpy())

  
            val_f1 = f1_score(val_labels, val_preds, average="macro")
            avg_val_loss = val_loss / len(val_dataloader)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train F1: {train_f1:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")

            # **Early Stopping Check**
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                best_model_state = self.model.state_dict()  # Save best model
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Stopping early after {epoch+1} epochs (no improvement for {patience} epochs).")
                    break  # Stop

        # Restore best model after early stopping
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)