import numpy as np

class RandomSolver():
    """
    A solver that just uses random sampling to perform instance selection

    Attributes:
        X (array-like): The input data
        Y (array-like): The target data
    """

    def __init__(self, X, Y, percentage_keep=0.75):
        self.X = X
        self.Y = Y
        self.percentage_keep = percentage_keep
 
    def run_solver(self, **kwargs):
        """Randomly samples instances based on percentage_keep"""
        num_samples = int(len(self.X) * self.percentage_keep)        
        sampled_indices = np.random.choice(len(self.X), num_samples, replace=False)

        sampled_X = self.X[sampled_indices]
        sampled_Y = self.Y[sampled_indices]
        
        return sampled_X, sampled_Y