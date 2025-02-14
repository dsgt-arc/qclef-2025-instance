class QuantumBatch:
    def __init__(self, Xbatch, embedding=None, label="", docs_range=[], Ybatch=None):
        self.Xbatch = Xbatch
        self.embedding = embedding
        self.label = label
        self.docs_range = docs_range
        self.sampler=None
        self.bqm=None
        self.Ybatch = Ybatch