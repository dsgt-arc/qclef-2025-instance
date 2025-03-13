from src.models.BaselineModel import BaselineTrainer
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.metrics import f1_score
import numpy as np

class Evaluator:
    def __init__(self, orig_folds, is_folds):
        """
        :param folds: Complete list of (X_train, y_train, X_val, y_val, X_test, y_test) splits
        :param instance_selector: List of (X_train, y_train, X_val, y_val, X_test, y_test) splits 
            where X_train and y_train have already gone through the instance selector
        """
        self.orig_folds = orig_folds
        self.is_folds = is_folds
        self.model = None

    def size_reduction(self, orig_data, is_data):
        return (len(orig_data) - len(is_data))/len(orig_data)
    
    def train_model(self, X_train, y_train, num_labels=2, batch_size=4, epochs=3):
        """Trains the model on selected instances."""
        input_dim = X_train.shape[1]
        trainer = BaselineTrainer(X_train, y_train, input_dim=input_dim, num_labels=num_labels)
        trainer.train(batch_size=batch_size, epochs=epochs)
        self.model = trainer.model
        return self.model

    def evaluate_fold(self, X_test, y_test, batch_size=8):
        """Evaluates the trained model on a single test dataset."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        self.model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), 
                                torch.tensor(y_test, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=batch_size)

        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                embeddings, labels = batch[0].to(device), batch[1].to(device)

                outputs = self.model(embeddings)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        return f1_score(all_labels, all_preds, average="macro")

    def cross_validation(self, num_labels=2, batch_size=8, epochs=3):
        """Runs cross-validation by training a model on selected instances, and evaluating it."""
        f1_scores = []
        size_reductions = []

        for fold_idx, (X_train, y_train, X_val, y_val, X_test, y_test) in enumerate(self.is_folds):
            print(f"Processing Fold {fold_idx+1}/{len(self.is_folds)}...")

            # Calculate Size reduction
            size_reduction = self.size_reduction(self.orig_folds[fold_idx][0], self.is_folds[fold_idx][0])
            size_reductions.append(size_reduction)

            # Train model
            self.train_model(X_train, y_train, num_labels, batch_size, epochs)

            # Evaluate model
            macro_f1 = self.evaluate_fold(X_val, y_val, batch_size)
            f1_scores.append(macro_f1)

            print(f"Fold {fold_idx+1}: F1 Score = {macro_f1:.4f}")

        return {
            "avg_f1": np.mean(f1_scores),
            "std_f1": np.std(f1_scores),
            "avg_size_reduction": np.mean(size_reductions)
        }
