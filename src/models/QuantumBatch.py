class QuantumBatch:
    def __init__(self, Xbatch, embedding=None, label="", docs_range=[], Ybatch=None):
        """Batch of data with X,Y, embedding (on the QPU), range in the original dataset, sampler type, binary quadratic model formulation
        """
        self.Xbatch = Xbatch
        self.embedding = embedding
        self.label = label
        self.docs_range = docs_range
        self.sampler=None
        self.bqm=None
        self.Ybatch = Ybatch