import numpy as np
from numba import jit

@jit(nopython=True)
def to_chaos_game(sequence, resolution, nucleotide_mapping, mode='FCGR'):
    """
    Generate the chaos game representation of a DNA sequence.

    Parameters:
    - sequence (str): DNA sequence (composed of 'A', 'C', 'G', 'T').
    - resolution (int): The resolution of the output image (e.g., 512 for a 512x512 image).
    - nucleotide_mapping (list): Coordinates for 'A', 'C', 'G', 'T' in a tuple format.
    - mode (str): 'FCGR' for Frequency Chaos Game Representation (increment) or 'BCGR' for Binary Chaos Game Representation (set to 1).

    Returns:
    - np.array: Flattened array representing the chaos game image.
    """
    
    image = np.zeros((resolution, resolution), dtype=np.uint8)

    x, y = 0.5, 0.5
    scale = resolution - 1

    for char in sequence:
        if char == 'A':
            index = 0
        elif char == 'C':
            index = 1
        elif char == 'G':
            index = 2
        elif char == 'T':
            index = 3
        else:
            continue  # Skip unknown characters

        corner_x, corner_y = nucleotide_mapping[index]
        x = (x + corner_x) / 2
        y = (y + corner_y) / 2

        ix, iy = int(x * scale), int(y * scale)
        
        if mode == 'FCGR':
            image[iy, ix] += 1
        elif mode == 'BCGR':
            image[iy, ix] = 1

    return image.flatten()

def chaos_game_representation(partition, resolution, mode='FCGR'):
    """
    Apply the chaos game representation to a partition of DNA sequences.

    Parameters:
    - partition (DataFrame): DataFrame with a column 'Sequence' containing DNA sequences.
    - resolution (int): Resolution of the output image for each sequence.
    - mode (str): 'FCGR' for Frequency Chaos Game Representation or 'BCGR' for Binary Chaos Game Representation.

    Returns:
    - np.array: Array containing the chaos game representations for each sequence.
    """
    nucleotide_mapping = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32)
    
    local_features = np.zeros((len(partition), resolution * resolution), dtype=np.uint8)
    for i, sequence in enumerate(partition['Sequence']):
        local_features[i, :] = to_chaos_game(sequence, resolution, nucleotide_mapping, mode)
    return local_features