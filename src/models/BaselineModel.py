import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.metrics import f1_score
from typing import List, Dict
import random
from torch.nn.utils.rnn import pad_sequence

class CustomEmbeddingModel(torch.nn.Module):
    """Custom model to use with pre-computed embeddings"""
    def __init__(self, input_dim, num_labels):
        super(CustomEmbeddingModel, self).__init__()
        # model layer for classification on embeddings
        self.fc = torch.nn.Linear(input_dim, num_labels)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, embeddings):
        # embeddings shape: (batch_size, embedding_dim)
        x = self.fc(embeddings)
        return self.softmax(x)

class TokenizedDataset(Dataset):
    """PyTorch dataset for pre-tokenized text samples."""
    def __init__(self, data: List[Dict]):
        """
        Args:
            data: List of dictionaries with 'input_ids' (embeddings) and 'label' keys.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.data[idx]["input_ids"], dtype=torch.float32),
            "label": torch.tensor(self.data[idx]["label"], dtype=torch.long),
        }

def collate_fn(batch):
    """Collate function to dynamically pad sequences (if needed)."""
    input_ids = [item["input_ids"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    input_ids = torch.stack(input_ids)

    return {"input_ids": input_ids, "label": labels}

class RandomSamplingTrainer:
    """Trainer for fine-tuning transformer models using random instance selection."""
    def __init__(self, input_dim: int, num_labels: int, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CustomEmbeddingModel(input_dim, num_labels).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()

    def random_sample(self, data: List[Dict], sample_ratio: float) -> List[Dict]:
        """Randomly samples a subset of instances."""
        sample_size = int(sample_ratio * len(data))
        return random.sample(data, sample_size)

    def train(self, data: List[Dict], sample_ratio: float, batch_size: int = 8, epochs: int = 3, lr: float = 5e-5):
        """
        Runs training with random sampling.
        """
        sampled_data = self.random_sample(data, sample_ratio)

        # Create DataLoader with padding support (for if we use non pre-embedded datasets)
        dataset = TokenizedDataset(sampled_data)
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
