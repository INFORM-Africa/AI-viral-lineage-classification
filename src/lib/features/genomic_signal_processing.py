import numpy as np
import pyfftw

def antisymmetric_padding(seq, target_len):
    """
    Apply anti-symmetric padding to the sequence to reach the target length.

    Parameters:
    seq (list or np.array): Original sequence.
    target_len (int): Desired length of the padded sequence.

    Returns:
    np.array: Padded sequence.
    """
    seq = np.array(seq)
    seq_len = len(seq)
    pad_len = target_len - seq_len
    
    if pad_len > 0:
        pad_values = -seq[-pad_len:][::-1]
        padded_seq = np.concatenate([seq, pad_values])
    else:
        padded_seq = seq
    
    return padded_seq

def numeric_transform(sq, mapping):
    """
    Transform a nucleotide sequence into a numeric sequence based on a provided mapping.

    Parameters:
    - sequence (str): The nucleotide sequence (e.g., 'AGCT').
    - mapping (dict): Mapping from nucleotide characters to numeric values.

    Returns:
    - np.array: The numeric sequence as a NumPy array of floats.
    """
    
    # Create a NumPy array of the same length as the input sequence
    numSeq = np.zeros(len(sq), dtype=np.float32)

    # Map characters to indices: A->0, C->1, G->2, T->3, others->-1 (which will be ignored)
    char_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    indices = np.array([char_to_index.get(char, -1) for char in sq])

    # Use boolean indexing to set values only for valid indices
    valid_indices = (indices >= 0)
    numSeq[valid_indices] = mapping[indices[valid_indices]]

    return numSeq

def fourier_transform(sequence):
    """
    Compute the Fourier transform of a numeric sequence and return its real and imaginary components.

    Parameters:
    - sequence (np.array): Numeric sequence to transform.

    Returns:
    - np.array: Fourier transform of the sequence (complex numbers).
    """
    fft_object = pyfftw.builders.fft(sequence)
    fourier = fft_object()
    
    return np.abs(fourier)

def process_sequence(sequence, max_length, mapping):
    """
    Process a nucleotide sequence: transform to numeric, apply padding if necessary, and compute the Fourier magnitude spectrum.

    Parameters:
    - sequence (str): The nucleotide sequence to process.
    - max_length (int): The maximum length for padding or truncating the sequence.
    - mapping (dict): Mapping from nucleotide characters to numeric values.

    Returns:
    - np.array: Fourier magnitude spectrum of the processed sequence.
    """
    numeric_sequence = numeric_transform(sequence, mapping)

    if len(numeric_sequence) < max_length:
        numeric_sequence = antisymmetric_padding(numeric_sequence, max_length)
    else:
        numeric_sequence = numeric_sequence[:max_length]

    magnitude_spectrum = fourier_transform(numeric_sequence).astype(np.float32)

    return magnitude_spectrum

def get_map(mapping):
    if mapping == "real":
        return np.array([-1.5, 0.5, -0.5, 1.5], dtype=np.float32)  # A, C, G, T mapping in order
    elif mapping == "pp":
        return np.array([-1, 1, -1, 1], dtype=np.float32)  # A, C, G, T mapping in order
    elif mapping == "justa":
        return np.array([1, 0, 0, 0], dtype=np.float32)  # A, C, G, T mapping in order
    elif mapping == "eiip":
        return np.array([0.1260, 0.1340, 0.0806, 0.1335], dtype=np.float32)

def genomic_signal_representation(partition, median_length, mapping):
    partition = partition.reset_index()

    numeric_map = get_map(mapping)
    
    local_features = np.zeros((len(partition), median_length), dtype=np.float32)
    for i in range(len(partition)):
        local_features[i] = process_sequence(partition["Sequence"][i], median_length, numeric_map)
    
    return local_features

def pearson_correlation_dissimilarity(x, y):
    """
    Compute the Pearson correlation dissimilarity matrix between two sets of vectors.

    Parameters
    ----------
    x : np.array
      Shape N x L.

    y : np.array
      Shape M x L.

    Returns
    -------
    np.array
      N x M array where each element is a Pearson correlation dissimilarity.
    """
    # Ensure x and y have the same number of columns
    if x.shape[1] != y.shape[1]:
        raise ValueError('x and y must have the same number of columns.')

    # Compute the mean and standard deviation of each row
    mu_x = x.mean(axis=1, keepdims=True)
    mu_y = y.mean(axis=1, keepdims=True)
    s_x = x.std(axis=1, ddof=0, keepdims=True)
    s_y = y.std(axis=1, ddof=0, keepdims=True)

    # Center the vectors by subtracting the mean
    x_centered = x - mu_x
    y_centered = y - mu_y
    
    # Compute the covariance matrix
    cov = np.dot(x_centered, y_centered.T) / x.shape[1]
    
    # Normalize the covariance to get the Pearson correlation coefficient
    correlation = cov / np.dot(s_x, s_y.T)
    
    # Convert Pearson correlation coefficient to dissimilarity
    dissimilarity = (1 - correlation) / 2
    
    return dissimilarity.astype(np.float16)