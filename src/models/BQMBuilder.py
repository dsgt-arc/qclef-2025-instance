from abc import ABC, abstractmethod
from numpy.linalg import norm
import numpy as np
import dimod

class BQMBuilder(ABC):
    """
    Abstract base class for constructing Binary Quadratic Models (BQMs) from batches of data.

    This class defines the structure for building QUBO (Quadratic Unconstrained Binary Optimization)
    matrices and converting them into BQMs, which are useful for solving combinatorial optimization
    problems using quantum or classical annealing methods.

    Attributes:
        batch (QuantumBatch): The batch of data for which the QUBO matrix is constructed.
    """

    def __init__(self, batch):
        self.batch = batch
    
    @abstractmethod
    def _build_q_matrix(self):
        """
        Abstract method for constructing the QUBO matrix from the batch.

        Returns:
            np.ndarray: The constructed QUBO matrix.
        """
        pass

    @abstractmethod
    def _find_k(self):
        """
        Abstract method to determine the value of k, i.e., the number of non-zero elements 
        in the optimization problem.

        Returns:
            int: The value of k.
        """
        pass

    def _build_bqm(self, k):
        """
        Constructs a Binary Quadratic Model (BQM) from the computed QUBO matrix.

        Args:
            k (int): The number of non-zero elements to be included in the solution.

        Returns:
            dimod.BinaryQuadraticModel: The formulated BQM.
        """
        qubo = self._build_q_matrix()
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo, offset=0.0)
        penalty = self.maximum_energy_delta(bqm)

        # Add k-element constraint to the BQM
        kbqm = dimod.generators.combinations(bqm.variables, k, strength=penalty)
        kbqm.update(bqm)
       
        return kbqm
    
    def maximum_energy_delta(self, bqm):
        """
        Computes a conservative bound on the maximum change in energy when flipping a single variable.

        Args:
            bqm (dimod.BinaryQuadraticModel): The BQM for which the energy change is computed.

        Returns:
            float: The maximum change in energy when flipping a single variable.
        """
        return max(
            abs(bqm.get_linear(i)) +
            sum(abs(bqm.get_quadratic(i, j)) for j in (v for v, _ in bqm.iter_neighborhood(i)))
            for i in iter(bqm.variables)
        )

class BcosQmatPaper(BQMBuilder):
    """
    Implements the Bcos method for constructing QUBO matrices based on batch data.

    This class builds QUBO matrices using the cosine similarity of data points and 
    incorporates class balance constraints.

    Attributes:
        batch (QuantumBatch): The batch of data for which the QUBO matrix is constructed.
        percentage_keep (float): The fraction of variables to be kept in the optimization.
    """

    def __init__(self, batch, percentage_keep):
        super().__init__(batch)
        self.percentage_keep = percentage_keep
    
    def _find_k(self):
        """
        Determines the number of variables to be retained in the optimization.

        Returns:
            int: The computed value of k.
        """
        return int((self.batch.Xbatch.shape[0]) * self.percentage_keep)
    
    def _create_bqm(self):
        """
        Constructs the Binary Quadratic Model (BQM) for the given batch.

        Returns:
            dimod.BinaryQuadraticModel: The formulated BQM.
        """
        k = self._find_k()
        return self._build_bqm(k)

    def _class_balance_diagonal(self):
        """
        Computes the class balance coefficient for diagonal elements in the QUBO matrix.

        The balance coefficient adjusts the weight of diagonal elements to account for class imbalance
        in binary classification tasks.

        Returns:
            float: The computed class balance coefficient.
        """
        return (self.batch.Ybatch).sum() / self.batch.Ybatch.shape[0]
  
    def _bcos_off_diagonal(self):
        """
        Computes the off-diagonal elements of the QUBO matrix based on cosine similarity.

        Returns:
            np.ndarray: A matrix where each element represents the similarity between data points.
        """
        norms = norm(self.batch.Xbatch, axis=1, keepdims=True) 
        cosine_matrix = (self.batch.Xbatch @ self.batch.Xbatch.T) / (norms @ norms.T)

        Y_reshaped = self.batch.Ybatch.reshape(-1, 1)
        same_class = (Y_reshaped == Y_reshaped.T)

        # Apply the appropriate sign based on class similarity
        cosine_matrix = np.where(same_class, np.abs(cosine_matrix), -np.abs(cosine_matrix))
        
        return cosine_matrix
    
    def _build_q_matrix(self):
        """
        Constructs the QUBO matrix by combining diagonal and off-diagonal elements.

        Returns:
            np.ndarray: The constructed QUBO matrix.
        """
        Qmat = self._bcos_off_diagonal()
        
        # Compute and set class balance for diagonal elements
        for i in range(len(self.batch.Ybatch)):
            Qmat[i, i] = self._class_balance_diagonal()

        return Qmat
