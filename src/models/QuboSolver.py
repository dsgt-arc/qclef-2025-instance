import math 
import time
import numpy as np
 
from joblib import Parallel, delayed
from neal import SimulatedAnnealingSampler
from src.models.QuantumBatch import QuantumBatch
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system import LeapHybridSampler

from qclef import qa_access as qa
 
class QuboSolver():
    """
    A solver for Quadratic Unconstrained Binary Optimization (QUBO) problems using
    Simulated Annealing (SA) or other sampling techniques.

    Attributes:
        X (array-like): The input dataset.
        Y (array-like): The target dataset.
        batch_size (int): The size of each batch for processing (default is 80).
        sampler (str): The sampling method to use, default is 'SA' (Simulated Annealing).
        building_time (float or None): Tracks the time taken to build the QUBO matrices (default is None).
        annealing_time (float or None): Tracks the total time spent on annealing (default is None).
        cores (int): Number of CPU cores used for parallel processing (default is 12, from configuration).
        num_reads (int): Number of reads per batch when running the annealer (default is 200, from configuration).
    """

    def __init__(self, X, Y, batch_size=80, cores=12, num_reads=12, sampler = 'SA', percentage_keep=0.75, random_state=123):
        self.X = X
        self.Y = Y
        self.sampler = sampler
        self.batch_size = batch_size
        self.building_time = None
        self.annealing_time = None
        self.cores = cores #comes from config
        self.num_reads = num_reads #comes from config
        self.batch_size = batch_size
        self.percentage_keep = percentage_keep
        self.random_state = random_state
 
    def run_QuboSolver(self, qmat_method, **kwargs):
        # Create batches
        batches = self._split_in_batches(batch_size=self.batch_size)

        # Build Q mat (BQM) for each batch
        building_time_start = time.time()
        
        # This can be also parralelized if SA
        for batch in batches:
 
            bqm_model = qmat_method(batch, self.percentage_keep, self.Y.shape[0], **kwargs)
            batch.bqm = bqm_model._create_bqm()
            
        building_time_end = time.time() 
         
        if self.sampler=='QA':
            annealing_time_start = time.time()
            results = []
            for i in range(len(batches)):
                results_tmp = self.get_best_instances_qa(batches[i])
                results.append(results_tmp)

            annealing_time_end = time.time()
             
        if self.sampler=='SA':
            # Run the annealer
            
            # results = Parallel(n_jobs=self.cores)(delayed(self.get_best_instances_multiprocess_sa)(
            #             batch,
            #             num_reads=self.num_reads,
            #         ) for batch in batches)
            # annealing_time_start2 = time.time()
            # print(annealing_time_start2-annealing_time_start1)
            
            # annealing_time_start3 = time.time()
            
            annealing_time_start = time.time()
            results = []
            for i in range(len(batches)):
                results_tmp = self.get_best_instances_multiprocess_sa(batches[i])
                results.append(results_tmp)

            annealing_time_end = time.time()
        
        final_results = {}

        for res in results:
                final_results.update(res)
                
         # Collect the time for (i) Qmat formulation and for (ii) Annealing
        building_time = building_time_end - building_time_start
        annealing_time = annealing_time_end - annealing_time_start

        sampled_X = self.X.iloc[np.array(list(final_results.values()))==1, :]
        sampled_Y = self.Y[np.array(list(final_results.values()))==1]
    
        output = {
            'results': final_results,
            'annealing_time_total': annealing_time,
            'building_time': building_time,
            'sampled_X': sampled_X,
            'sampled_Y': sampled_Y
        }
        
        # Return the results and the time statistics
        return output

    def get_best_instances_qa(self, batch: QuantumBatch, i: int, num_reads=100):
        sampler = DWaveSampler(solver={'name': 'Advantage_system7.1'})
        
        kbqm = batch.bqm
        response = qa.submit(sampler, sampler.sample, kbqm, label=f'batch_{i}', num_reads=num_reads)
        final_response = {}
        for var, index in zip(batch.docs_range, sorted(response.first.sample.keys())):
            final_response[var] = response.first.sample[index]
            
          
    def get_best_instances_multiprocess_sa(self, batch: QuantumBatch, num_reads=100):
        """
        Samples from the batch using the Simulated Annealing algorithm 
        and returns immediately the response. This is done synchronously
        since there is no network involved and we can exploit parallelization
        to speed up this operation.
        TODO: Needs to be done multiprocess
        """
        sampler = SimulatedAnnealingSampler()
        
        response = self._run_sa_sampler(sampler, batch, num_reads)
        
        final_response = {}
        for var, index in zip(batch.docs_range, sorted(response.first.sample.keys())):
            final_response[var] = response.first.sample[index]
        
        return final_response

    def _run_sa_sampler(self, sampler, batch, num_reads):
        """
        Formulates the problem according to the batch and k provided and
        then samples it.
        """
        kbqm = batch.bqm
        response = sampler.sample(kbqm, label=batch.label, num_reads=num_reads)
        return response 

    def _split_in_batches(self, batch_size, start_index=0):
        """Splits the provided initial matrix into batches having size that can be at max
        the specified one.

        :param matrix: the initial matrix from which to derive the QUBO/BQM formulation
        :param batch_size: the maximum batch size
        :sampler_type: the type of sampler to use (e.g., Quantum or Simulated)
        :param percentage_kept: the percentage of instances to be kept
        :start_index: the starting index, useful when computing many matrices (e.g., in SmartHamQuboSolver)

        https://numpy.org/doc/stable/reference/generated/numpy.array_split.html 
        Nice function that splits taking care of the remainder


        """
        batches = []

        # The batches having one more samples if the division is not integer
        number_one_more = self.X.shape[0] % math.ceil(
            self.X.shape[0]/batch_size)
        
        X_splitted = np.array_split(
            self.X, math.ceil(self.X.shape[0]/batch_size), axis=0)
        Y_splitted = np.array_split(
            self.Y, math.ceil(self.Y.shape[0]/batch_size), axis=0)
        
        range_start = start_index
        range_end = start_index
 
        embedding_computed = None
        # ALL k BATCHES FIRST (FULL)
        for i in range(number_one_more):
            range_end += (X_splitted[i].shape[0])
            batch=QuantumBatch(
                Xbatch=X_splitted[i],
                embedding=embedding_computed,
                docs_range=np.arange(range_start, range_end),
                Ybatch=Y_splitted[i]
            )
            batches.append(batch)
            range_start += (X_splitted[i].shape[0])

            # Quantum part will come here for the embedding

        # LAST BATCH
        for i in range(number_one_more, len(X_splitted)):
            range_end += (X_splitted[i].shape[0])
            batch=QuantumBatch(
                Xbatch=X_splitted[i],
                embedding=embedding_computed,
                docs_range=np.arange(range_start, range_end),
                Ybatch =Y_splitted[i]
            )
            batches.append(batch)
            range_start += (X_splitted[i].shape[0])
        
        return batches
 
###########################################
   # UTILITIES - get them away

    # def plot_qubo_matrix(self, bqm, features):
    #     Q_np = np.ones((len(features), len(features)))*np.nan

    #     feature_to_index = {feature: index for index,
    #                         feature in enumerate(features)}
        
    #     for (i, j), value in bqm.to_qubo()[0].items():

    #         Q_np[i, j] = value
    #     plt.imshow(Q_np, cmap='jet')
    #     plt.colorbar()
    #     plt.show()

    # def plot_qubo_matrix_1(self, qubo, features):
    #     # Q_np = np.ones((len(features), len(features)))*np.nan
    #     # for (i, j), value in qubo.items():
    #     #     Q_np[i, j] = value

    #     #Q_np=np.tril(Q_np)
    #     #Q_np[np.triu_indices(Q_np.shape[0], 1)] = np.nan
        
    #     plt.imshow(qubo, cmap='jet')
    #     plt.colorbar()
    #     plt.show()

    
    
  