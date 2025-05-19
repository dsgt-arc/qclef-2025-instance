from abc import ABC, abstractmethod
from numpy.linalg import norm
import numpy as np
import dimod
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from joblib import Parallel, delayed

class BQMBuilder(ABC):
    """
    Abstract base class for constructing Binary Quadratic Models (BQMs) from batches of data.

    This class defines the structure for building QUBO (Quadratic Unconstrained Binary Optimization)
    matrices and converting them into BQMs, which are useful for solving combinatorial optimization
    problems using quantum or classical annealing methods.

    Attributes:
        batch (QuantumBatch): The batch of data for which the QUBO matrix is constructed.
    """

    def __init__(self, batch, X, y):
        self.batch = batch
        self.X = X
        self.y = y

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

    def __init__(self, batch, X, y, percentage_keep, sample_size):
        super().__init__(batch, X, y)
        self.percentage_keep = percentage_keep
        self.sample_size = sample_size    
    
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
        vals, counts = np.unique(self.batch.Ybatch, return_counts=True)
    
        diag_values = [counts[vals==i]/self.sample_size for i in self.batch.Ybatch]
 
        return diag_values  
  
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
     
    def _beuc_off_diagonal(self):
        """
        Computes the off-diagonal elements of the QUBO matrix based on Euclidean distance.

        Returns:
            np.ndarray: A matrix where each element represents the distance between data points.
        """
        # Compute raw Euclidean distances
        distance_matrix = cdist(self.batch.Xbatch, self.batch.Xbatch, metric='euclidean')

        Y_reshaped = self.batch.Ybatch.reshape(-1, 1)
        same_class = (Y_reshaped == Y_reshaped.T)

        # Apply the appropriate sign based on class similarity
        #distance_matrix = np.where(same_class, np.abs(distance_matrix), -np.abs(distance_matrix))
        distance_matrix = np.where(same_class, -np.abs(distance_matrix), np.abs(distance_matrix))
        return distance_matrix
        
    def _build_q_matrix(self):
        """
        Constructs the QUBO matrix by combining diagonal and off-diagonal elements.

        Returns:
            np.ndarray: The constructed QUBO matrix.
        """
        Qmat = self._bcos_off_diagonal()
        #Qmat = self._beuc_off_diagonal()
        # Compute and set class balance for diagonal elements
        
        normalized_counts = self._class_balance_diagonal()
        np.fill_diagonal(Qmat, normalized_counts)
 
        return Qmat
     
class SVC_diagonal(BQMBuilder):
    def __init__(self, batch, X, y, percentage_keep, sample_size):
        super().__init__(batch, X, y)
        self.percentage_keep = percentage_keep
        self.sample_size = sample_size
        self.distance_to_margin = None
        
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

    def _svc_distance_diagonal(self):
        return self._svc_margin_diagonal()   
    
    def _svc_margin_diagonal(self, kernel="rbf", C=1.0, gamma="scale",
                             inverse=True, normalise=True, eps=1e-12):
        """
        Uses an SVC as a *surrogate model* and turns each training
        point's margin distance into an influence score.

        Parameters
        ----------
        kernel, C, gamma : standard SVC hyper-parameters.
        inverse  : if True, score = 1 / (|margin| + eps)
                   else     score = -|margin|
        normalise: rescale scores to [0, 1] for numerical stability.
        eps      : small constant to avoid division by zero.

        Returns
        -------
        scores : ndarray, shape (n_samples,)
                 Higher  ⇒ more influential.
        """
        # X, y = self.X, self.y #for computing globally rather than in batches
        X, y = self.batch.Xbatch, self.batch.Ybatch

        svc = SVC(kernel=kernel, C=C, gamma=gamma, probability=False)
        svc.fit(X, y)

        margin_dist = np.abs(svc.decision_function(X))

        if inverse:
            scores = 1.0 / (margin_dist + eps)        # small margin should get bigger score

        if normalise:
            scores /= scores.max()

        return -scores
    
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
        #Qmat = self._beuc_off_diagonal()
        # Compute and set class balance for diagonal elements

        #for computing globally rather than by batches
        # if self.distance_to_margin is None:
        #     self.distance_to_margin = self._svc_distance_diagonal()

        # batch_indices = self.batch.docs_range
        # batch_scores = self.distance_to_margin[batch_indices]
        # np.fill_diagonal(Qmat, batch_scores)

        distance_to_margin = self._svc_distance_diagonal()
        np.fill_diagonal(Qmat, distance_to_margin)
        
        return Qmat
    
    
class IterativeDeletion(BQMBuilder):
    """
    This class builds QUBO matrices using the cook's distance of data points found by 
    iteratively deleting instances and retraining the model.

    Attributes:
        batch (QuantumBatch): The batch of data for which the QUBO matrix is constructed.
        percentage_keep (float): The fraction of variables to be kept in the optimization.
    """

    def __init__(self, batch, X, y, percentage_keep, sample_size):
        super().__init__(batch, X, y)
        self.percentage_keep = percentage_keep
        self.sample_size = sample_size
        self.influence_scores = None
    
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

    def _influence_scores_diagonal(self):
        """
        Computes influence scores for use as diagonal values in the QUBO matrix.

        Returns:
            np.ndarray: Influence scores using cook's distance numerator
        """
        influence_scores = self._compute_influence_logistic()

        return influence_scores
    
    def _compute_influence_logistic(self, target_class=1, cv=5):
        """
        Computes influence scores for training instances using logistic regression.
        
        Parameters:
            target_class (int): The class for which to measure influence.
            cv (int): Number of folds for cross-validation in logistic regression.

        Returns:
            influence_scores (ndarray): Influence scores for each training instance.
        """
        
        # Train logistic regression with automatic hyperparameter tuning
        # X, Y = self.X, self.y #for computing globally rather than by batches

        X, Y = self.batch.Xbatch, self.batch.Ybatch
        n = len(X)
        baseline = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
        baseline.fit(X, Y)
        
        baseline_probs = baseline.predict_proba(X)[:, target_class]
       
        influence_scores = np.empty(n)

        # Leave-one-out retraining

        influence_scores = Parallel(n_jobs=-1)(
            delayed(self._influence_logistic)(i, X, Y, baseline_probs, target_class)
            for i in range(n)
        )
 
        return -np.array(influence_scores)
    
    def _influence_logistic(self, i, X, y, baseline_probs, target_class):
        mask = np.ones(len(y), dtype=bool)
        mask[i] = False
        X_sub, y_sub = X[mask], y[mask]
        model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
        model.fit(X_sub, y_sub)
        new_probs = model.predict_proba(X)[:, target_class]
        return np.mean(np.abs(new_probs - baseline_probs))
  
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
       
        # For if we computed global scores
        # if self.influence_scores is None:
        #     self.influence_scores = self._influence_scores_diagonal()
        
        # batch_indices = self.batch.docs_range
        # batch_scores = self.influence_scores[batch_indices]
        
        # np.fill_diagonal(Qmat, batch_scores)

        influence_scores = self._influence_scores_diagonal()
        np.fill_diagonal(Qmat, influence_scores)
 
        return Qmat
    
class SVC_iterative_combined(BQMBuilder):
    """
    This class builds QUBO matrices using both the iterative deletion method and the SVC method 
    for the diagonal, with bcos on the off-diaagonal.

    Attributes:
        batch (QuantumBatch): The batch of data for which the QUBO matrix is constructed.
        percentage_keep (float): The fraction of variables to be kept in the optimization.
    """

    def __init__(self, batch, X, y, percentage_keep, sample_size):
        super().__init__(batch, X, y)
        self.iterative_deletion = IterativeDeletion(batch, X, y, percentage_keep, sample_size)
        self.svc_diagonal = SVC_diagonal(batch, X, y, percentage_keep, sample_size)
        self.percentage_keep = percentage_keep
        self.sample_size = sample_size
        self.svc_scores = None
        self.iterative_scores = None
        self.combined_scores = None
    
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

    def _combined_diagonal(self, alpha):
        """
        Computes the combined score for cooks distance (iterative deletion) + SVC.

        Parameters:
            alpha (float): Variable to determine the weight of SVC scores. 1-alpha determines the 
            weight for the Iterative Deletion scores.

        Returns:
            float: The computed combined influence scores.
        """
        scaler = MinMaxScaler()

        svc_scores = self.svc_diagonal._svc_distance_diagonal().reshape(-1, 1)
        cooks_scores = self.iterative_deletion._influence_scores_diagonal().reshape(-1, 1)

        svc_scaled = scaler.fit_transform(svc_scores).flatten()
        cooks_scaled = scaler.fit_transform(cooks_scores).flatten()

        combined_influence_scores = alpha * svc_scaled + (1 - alpha) * cooks_scaled

        return combined_influence_scores
     
    def _build_q_matrix(self, alpha=0.5):
        """
        Use cosine similarity off-diagonal and combined SVC+Cook's diagonal.
        """
        Qmat = self.iterative_deletion._bcos_off_diagonal()

        # For if we did global scores
        # if self.combined_scores is None:
        #     self.combined_scores = self._combined_diagonal(alpha)
        
        # np.fill_diagonal(Qmat, self.combined_scores)

        np.fill_diagonal(Qmat, self._combined_diagonal(alpha))

        return Qmat
