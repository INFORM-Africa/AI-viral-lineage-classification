"""
This is the implementation of the feature extraction technique proposed in 
Coutinho, M.G.F., Câmara, G.B.M., Barbosa, R.D.M., Fernandes.
SARS-CoV-2 virus classification based on stacked sparse autoencoder. 
Computational and Structural Biotechnology Journal 21, 284–298, M.A.C., 2023.
doi: https://doi.org/10.1016/j.csbj.2022.12.007
"""

import itertools
import collections
import numpy as np
from typing import Iterable

class CoutinhoFeatures:
    def __init__(self, k:int=7, resolution:int=None):
        if resolution is not None:
            self.k = int(np.log2(resolution))
        else:
            self.k = k
            
        self.L = 2**k
        self.kmers = [''.join(kmer) for kmer in itertools.product(['A', 'C', 'T', 'G'], repeat=k)]
    
    def extract(self, sequences:Iterable[str], b:int=8) -> np.ndarray:
        features = [self.extract_single(seq=sequence, b=b) for sequence in sequences]
        return np.array(features)

    def extract_single(self, seq:str, b:int=8) -> np.ndarray:
        sub_seqs = [seq[i : i + self.k] for i in range(len(seq) - self.k + 1)]
        kmer_counts = collections.Counter(sub_seqs)
        features = np.array([kmer_counts[kmer] for kmer in self.kmers]).reshape((self.L, self.L))
        features =  (2**b - 1) * features / np.max(features)
        return features
