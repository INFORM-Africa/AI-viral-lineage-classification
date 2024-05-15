import itertools
import numpy as np
from typing import Iterable, Literal
from concurrent.futures import ProcessPoolExecutor

class ConventionalFeatures:
    def __init__(self):
        pass
    
    def extract_binary_features(self, sequences:Iterable[str]) -> np.ndarray:
        features = [self.extract_binary_features_single(sequence) for sequence in sequences]
        return np.stack(features)

    @staticmethod
    def extract_binary_features_single(sequence:str) -> np.ndarray:
        
        sequence_arr = np.array(list(sequence))

        features = np.column_stack([
            np.where(sequence_arr == 'A', 1, 0),
            np.where(sequence_arr == 'C', 1, 0),
            np.where(sequence_arr == 'T', 1, 0),
            np.where(sequence_arr == 'G', 1, 0)
        ])
        features = features.T

        return features

    def extract_kmers_features(self, sequences:Iterable[str], k:int, normalize:bool=True) -> np.ndarray:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.extract_kmers_features_single, seq,  k, normalize) for seq in sequences]
            
        features = [f.result() for f in futures]
        return np.array(features)

    @staticmethod
    def extract_kmers_features_single(sequence: str, k:int, normalize: bool) -> np.ndarray:
        bases = ['A', 'C', 'T', 'G']
        kmers = [''.join(kmer) for kmer in itertools.product(bases, repeat=k)]
        kmers_dict = {kmer: 0 for kmer in kmers}

        L = len(sequence) - k + 1
        for i in range(L):
            kmer = sequence[i:i + k]
            if 'N' not in kmer:
                kmers_dict[kmer] += 1

        # keys = np.array([kmer for kmer in kmers_dict])
        values = np.array([kmers_dict[kmer] for kmer in kmers])

        if normalize:
            values = values / sum(values)

        return values 
    
    def extract_fcgr_features(self, sequences:Iterable[str], resolution:Literal[64, 128, 256]=128) -> np.ndarray:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.extract_fcgr_features_single, seq, resolution) for seq in sequences]
    
        features = [f.result() for f in futures]       
        return np.array(features)

    @staticmethod
    def extract_fcgr_features_single(sequence:str, resolution:Literal[64, 128, 256]) -> np.ndarray:
        coordinates = {
            'A': (0, 0),
            'C': (0, 1),
            'G': (1, 1),
            'T': (1, 0),
        }
        fcgr = np.zeros((resolution, resolution), dtype=np.int16)

        x, y = 0.5, 0.5
        scale = resolution - 1

        for base in sequence:
            if base not in coordinates:
                continue

            corner_x, corner_y = coordinates[base]
            x = (x + corner_x) / 2
            y = (y + corner_y) / 2

            ix, iy = int(x * scale), int(y * scale)
            fcgr[iy, ix] = fcgr[iy, ix] + 1

        return fcgr
