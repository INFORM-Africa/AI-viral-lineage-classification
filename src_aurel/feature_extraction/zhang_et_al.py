"""
This is the implementation of the feature extraction technique proposed in 
Zhang, Chun-Ting & Zhang, Ren & Ou, Hong-Yu. (2003). 
The Z curve database: A graphic representation of genome sequences. 
Bioinformatics (Oxford, England). 19. 593-9. 
doi: http://dx.doi.org/10.1093/bioinformatics/btg041
"""
import numpy as np
from typing import Iterable
from timebudget import timebudget

class ZhangFeatures:
    def __init__(self):
        pass
    
    def extract(self, sequences:Iterable[str], smooth:bool=False) -> np.ndarray:
        features = [self.extract_single(seq=sequence, s=smooth) for sequence in sequences]
        return np.array(features)

    @timebudget
    def extract_single(self, seq:str, smooth:bool=False) -> np.ndarray:
        sequence_arr = np.array(list(seq))

        An = np.cumsum(sequence_arr == 'A')
        Cn = np.cumsum(sequence_arr == 'C')
        Gn = np.cumsum(sequence_arr == 'G')
        Tn = np.cumsum(sequence_arr == 'T')

        x_n = An + Gn - Cn - Tn
        y_n = An + Cn - Gn - Tn
        z_n = An + Tn - Gn - Cn

        features = np.column_stack([x_n, y_n, z_n])

        if smooth:
            features = self._smooth_coordinates(features)

        return features

    @staticmethod
    def _smooth_coordinates(features:np.ndarray) -> np.ndarray:
        padded_x = np.pad(features[:, 0], (1, 1), mode='edge')
        padded_y = np.pad(features[:, 1], (1, 1), mode='edge')
        padded_z = np.pad(features[:, 2], (1, 1), mode='edge')

        smoothed_x = (1/6) * padded_x[:-2] + (2/3) * padded_x[1:-1] + (1/6) * padded_x[2:]
        smoothed_y = (1/6) * padded_y[:-2] + (2/3) * padded_y[1:-1] + (1/6) * padded_y[2:]
        smoothed_z = (1/6) * padded_z[:-2] + (2/3) * padded_z[1:-1] + (1/6) * padded_z[2:]

        smoothed_features = np.column_stack([smoothed_x, smoothed_y, smoothed_z])

        return smoothed_features