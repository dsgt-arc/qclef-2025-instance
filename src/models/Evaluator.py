from src.models.BaselineModel import BaselineTrainer
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import f1_score

class Evaluator:
    def __init__(self, data, X_results, y_results):
        """
        :param data: Full dataset
        :param results: Instance selected dataset returned by the solver
        """
        self.data = data
        self.X_results = X_results
        self.y_results = y_results
        self.model = None

    def size_reduction(self):
        """Calculates the percentage reduction in sample size"""
        
        return (self.X_results.shape[0]) / self.data[0][0].shape[0]
    
    def train_model(self, num_labels=2, batch_size = 4, epochs = 3):
        input_dim = self.X_results.shape[1]
        trainer = BaselineTrainer(self.X_results, self.y_results, input_dim=input_dim, num_labels=num_labels)
        trainer.train(batch_size=batch_size, epochs=epochs)

        self.model = trainer.model

        return
    
    def evaluate(self, X_test_data, y_test_data, batch_size=8):
        """
        Evaluates the trained model on a test dataset and returns macro F1 score.
        
        :param test_data: List of dictionaries with 'input_ids' and 'label' keys
        :param batch_size: Batch size for evaluation
        :return: Macro F1 score
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        self.model.eval()  # Set to evaluation mode
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        dataset = (X_test_data, y_test_data)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                embeddings = batch["input_ids"].to(device)
                labels = batch["label"].to(device)

                outputs = self.model(embeddings)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        return macro_f1

