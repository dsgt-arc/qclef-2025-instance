from abc import ABC, abstractmethod
from numpy.linalg import norm
import numpy as np
import dimod

# Generic class that builds a Q mat for a batch depending on the method
class BQMBuilder(ABC):
    def __init__(self, batch):
        self.batch = batch
    
    @abstractmethod
    def _build_q_matrix(self):
        # Method that builds the q matrix
        pass

    @abstractmethod
    def _find_k(self):
        # Method that sets the k (i.e. number of non 0s) searched
        pass

    def _build_bqm(self, k):
        
        qubo = self._build_q_matrix()
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo, offset=0.0)
        penalty = self.maximum_energy_delta(bqm)

        # This here is critical (to add the k), see the documentation
        kbqm = dimod.generators.combinations(
            bqm.variables, k, strength=penalty)
        kbqm.update(bqm)
       
        return kbqm
    
    # WHAT DOES THIS DO?
    def maximum_energy_delta(self, bqm):
        """Computes conservative bound on maximum change in energy when flipping a single variable"""
        return max(abs(bqm.get_linear(i))
                   + sum(abs(bqm.get_quadratic(i, j))
                         for j in (v for v, _ in bqm.iter_neighborhood(i)))
                   for i in iter(bqm.variables))
 
    
# BCos method for building Q mat from a batch. Inherts from QmatBuilder.
class BcosQmatPaper(BQMBuilder):
    def __init__(self, batch, percentage_keep):
        super().__init__(batch)
        self.percentage_keep=percentage_keep
    
    def _find_k(self):
        return int((self.batch.Xbatch.shape[0])*(self.percentage_keep))
    
    def _create_bqm(self):
        k = self._find_k()
        bqm = self._build_bqm(k)
        return bqm

    # Diagonal factor in the Bcos Solution
    def _class_balance_diagonal(self):
        """
            Computes the balancing factor of the label of the class for two classes use case
        """
        coefficient = (self.batch.Ybatch).sum()/self.batch.Ybatch.shape[0] 
        return coefficient
  
    def _bcos_off_diagonal(self):
        """
        """
        norms = norm(self.batch.Xbatch, axis=1, keepdims=True) 
        cosine_matrix = (self.batch.Xbatch @ self.batch.Xbatch.T) / (norms @ norms.T)
 
        Y_reshaped = self.batch.Ybatch.reshape(-1, 1)       
        same_class = (Y_reshaped == Y_reshaped.T)
        
        # apply the positive/negative sign
        cosine_matrix = np.where(same_class, np.abs(cosine_matrix), -np.abs(cosine_matrix))
        
        return cosine_matrix
    
    # This computers the off diagonal + adds the diagonal
    def _build_q_matrix(self):
        """ Paper of Pasin method"""

        Qmat = self._bcos_off_diagonal()
        # Compute class balance for the diagonal elements
        for i in range(len(self.batch.Ybatch)):
            Qmat[i, i] = self._class_balance_diagonal()

        return Qmat
    


   
