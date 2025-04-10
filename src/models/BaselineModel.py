import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.metrics import f1_score
from typing import List, Dict
import random
from torch.nn.utils.rnn import pad_sequence

class TokenizedDataset(Dataset):
    """PyTorch dataset for pre-tokenized text samples.
    
    Attributes:
        data: List of dictionaries with 'input_ids' (embeddings) and 'label' keys.
    """
    def __init__(self, texts, labels, tokenizer, max_length=256):
        
        self.X = texts
        self.y = labels
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.X[idx], 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0), 
            "attention_mask": encoding["attention_mask"].squeeze(0), 
            "labels": torch.tensor(self.y[idx], dtype=torch.long)
        }

def collate_fn(batch):
    """Collate function to dynamically batch tokenized sequences."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask, 
        "labels": labels
    }

# Load and compile our model
bert_model = "distilbert-base-uncased"

class BaselineTrainer:
    """Trainer for the baseline model

    Attributes:
        data: The reduced dataset after performing instacne selection
        """
    def __init__(self, X_train, y_train, X_val, y_val, model, tokenizer, input_dim: int, num_labels: int, device: str = None):
        self.X_train = [str(text) for text in X_train]
        self.y_train = y_train
        self.X_val = [str(text) for text in X_val]
        self.y_val = y_val
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, batch_size: int = 8, epochs: int = 3, lr: float = 5e-5, patience=5):
        """
        Runs the training loop for the baseline model
        """

        # Create DataLoader
        dataset_train = TokenizedDataset(texts = self.X_train, labels = self.y_train, tokenizer = self.tokenizer)
        dataset_val = TokenizedDataset(texts = self.X_val, labels = self.y_val, tokenizer = self.tokenizer)
        
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
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()

                # Feed embeddings to the model
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
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
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = self.criterion(outputs.logits, labels)
                    val_loss += loss.item()

                    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
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