import numpy as np
import time

class RandomSolver():
    """
    A solver that just uses random sampling to perform instance selection

    Attributes:
        X (array-like): The input data
        Y (array-like): The target data
        solving_time: Time taken to take the random instances
    """

    def __init__(self, X, Y, percentage_keep=0.75):
        self.X = X
        self.Y = Y
        self.percentage_keep = percentage_keep
 
    def run_solver(self, **kwargs):
        """Randomly samples instances based on percentage_keep"""
        solving_time_start = time.time()
        
        num_samples = int(len(self.X) * self.percentage_keep)        
        sampled_indices = np.random.choice(len(self.X), num_samples, replace=False)

        sampled_X = self.X[sampled_indices]
        sampled_Y = self.Y[sampled_indices]

        solving_time_end = time.time()

        self.solving_time = solving_time_end - solving_time_start
        
        return sampled_X, sampled_Y