from src.models.BaselineModel import BaselineTrainer, TokenizedDataset
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import f1_score
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

class Evaluator:
    def __init__(self, orig_folds, is_folds_raw, is_folds_embed, config, bert_model: str ="distilbert-base-uncased", num_labels: int = 2, device: str = None, model_type: str = None):
        """
        :param folds: Complete list of (X_train, y_train, X_val, y_val, X_test, y_test) splits
        :param instance_selector: List of (X_train, y_train, X_val, y_val, X_test, y_test) splits 
            where X_train and y_train have already gone through the instance selector
        """
        self.orig_folds = orig_folds
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.bert_model = bert_model
        self.num_labels = num_labels
        self.config = config
        self.model_type = model_type

        if model_type=="bert":
            self.is_folds = is_folds_raw
        else:
            self.is_folds = is_folds_embed

    def size_reduction(self, orig_data, is_data):
        return (len(orig_data) - len(is_data))/len(orig_data)
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Trains the model on selected instances."""

        if self.model_type=="bert":        
            input_dim = X_train.shape[1]
            model = AutoModelForSequenceClassification.from_pretrained(self.bert_model, num_labels=self.num_labels).to(self.device)
            trainer = BaselineTrainer(X_train, y_train, X_val, y_val, model, self.tokenizer, input_dim=input_dim, num_labels=self.num_labels)
            trainer.train(**self.config.network_model)
            return trainer.model

        elif self.model_type=="logreg":
            model = LogisticRegression(solver='liblinear', random_state=0)
            model.fit(X_train, y_train)
            return model
        
        else:
            print("Incorrect model_type entered")
            return None
        
    def evaluate_fold(self, model, X_test, y_test, batch_size=8):
        """Evaluates the trained model on a single test dataset."""

        if self.model_type=="bert":
            model.eval()
            model.to(self.device)

            X_test = [str(text) for text in X_test]

            dataset = TokenizedDataset(texts = X_test, labels = y_test, tokenizer = self.tokenizer)
            dataloader = DataLoader(dataset, batch_size=batch_size)

            all_preds, all_labels = [], []

            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())

            return f1_score(all_labels, all_preds, average="macro")

        elif self.model_type=="logreg":
            all_preds = model.predict(X_test)
            all_labels = y_test

            return f1_score(all_labels, all_preds, average="macro")
        
        else:
            print("Incorrect model_type entered")
            return None

    def cross_validation(self, num_labels=2):
        """Runs cross-validation by training a model on selected instances, and evaluating it."""
    
        models = []
        f1_scores = []
        size_reductions = []    
        
        for fold_idx, (X_train, y_train, X_val, y_val, X_test, y_test) in enumerate(self.is_folds):
            
            print(f"Processing Fold {fold_idx+1}/{len(self.is_folds)}...")

            # Calculate Size reduction
            size_reduction = self.size_reduction(self.orig_folds[fold_idx][0], self.is_folds[fold_idx][0])
            size_reductions.append(size_reduction)

            # Train model
            model = self.train_model(X_train, y_train, X_val, y_val)
            models.append(model)

            # Evaluate model
            macro_f1 = self.evaluate_fold(model, X_test, y_test, self.config.network_model.batch_size)
            f1_scores.append(macro_f1)

            print(f"Fold {fold_idx+1}: F1 Score = {macro_f1:.4f}")

        return {
            "avg_f1": np.mean(f1_scores),
            "std_f1": np.std(f1_scores),
            "avg_size_reduction": np.mean(size_reductions)
        }
