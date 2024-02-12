#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import pandas as pd
from numba import njit
import time
from pydivsufsort import divsufsort, kasai, common_substrings
import time
import numba


# In[2]:
import threading
import time
import numpy as np

# Initialize a lock
lock = threading.Lock()

def safe_divsufsort_kasai(S):
    with lock:
        # Only one thread can enter this block at a time
        SA = get_SA(S)  # Assuming divsufsort is an external, optimized function
    return SA

@numba.njit
def process_first_loop(LCP, same_seq, n):
    f = np.zeros(n, dtype=np.int16)  # Initialize the result array f with zeros
    min_val = 0  # Initialize min to 0
    for i in range(1, n-2):
        if same_seq[i]:
            if LCP[i+1] < min_val:
                min_val = LCP[i+1]
            f[i + 1] = min_val
        else:
            min_val = LCP[i+1]
            f[i + 1] = LCP[i+1]
    return f

@numba.njit
def process_second_loop(LCP, same_seq, n):
    f = np.zeros(n, dtype=np.int16)  # Initialize the result array f with zeros
    min_val = 0  # Re-initialize min to 0 for the second loop
    for i in range(n-1, 1, -1):
        if same_seq[i-1]:  # Adjusted index for same_seq
            if LCP[i] < min_val:
                min_val = LCP[i]
            f[i - 1] = max(min_val, f[i - 1])
        else:
            min_val = LCP[i]
            f[i - 1] = max(min_val, f[i - 1])
    return f

def get_SA(S):
    SA = divsufsort(S)  # Assuming divsufsort is an external, optimized function
    return SA

def acs(A, B):
    start_time = time.time()

    S = f"{A}${B}"

    SA = safe_divsufsort_kasai(S)
    
    LCP = kasai(S, SA)
    LCP = np.append(-1, LCP[:-1])  # Assuming kasai is an external, optimized function

    n = len(S)
    mid = len(A) + 1

    # Compute same_seq array
    is_A = SA < mid
    is_A_shifted = np.roll(is_A, -1)
    same_seq = is_A & is_A_shifted
    
    SA = get_SA(S)  # Assuming divsufsort is an external, optimized function

    kasai_start = time.time()
    LCP = kasai(S, SA)
    LCP = np.append(-1, LCP[:-1])# Assuming kasai is an external, optimized function

    same_seq |= (~is_A) & np.roll(~is_A, -1)
    same_seq_end = time.time()
    
    same_seq[-1] = False

    f1 = process_first_loop(LCP, same_seq, n)

    f2 = process_second_loop(LCP, same_seq, n)

    # Merge the results from both loops
    f = np.maximum(f1, f2)
    
    A_scores = f[is_A]
    B_scores = f[~is_A]
    
    d_AB = np.log(len(B))/np.mean(A_scores) - 2*np.log(len(A))/len(A)
    d_BA = np.log(len(A))/np.mean(B_scores) - 2*np.log(len(B))/len(B)
    
    d_ACS = (d_AB + d_BA)/2
    return d_ACS


# In[3]:


data = pd.read_parquet('../../data/processed/genomes.parquet', engine='pyarrow')  # You can use 'fastparquet' as the engine
data


# In[4]:


data['Sequence'] = data['Sequence'].str.replace('[^ACTG]', '', regex=True)


# In[5]:


def random_undersample(data_df, max_samples_per_class=1, random_state=42):
        
    undersampled_data = []

    for class_value, group in data_df.groupby('Lineage'):
        if len(group) > max_samples_per_class:
            undersampled_group = group.sample(n=max_samples_per_class, random_state=random_state)
        else:
            undersampled_group = group
        undersampled_data.append(undersampled_group)

    undersampled_data_df = pd.concat(undersampled_data)
    undersampled_data_df = undersampled_data_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return undersampled_data_df


# In[6]:


comparison_sequences = random_undersample(data)[:50]["Sequence"]


import concurrent.futures
import time
import numpy as np

def calculate_acs_for_genome(genome):
    scores = np.zeros(len(comparison_sequences))
    for i, comp_seq in enumerate(comparison_sequences):
        scores[i] = acs(genome, comp_seq)
    return scores

start = time.time()

# Assuming `data["Sequence"]` contains your genomes and is a Pandas Series or similar iterable
# and `comparison_sequences` is a list of sequences you are comparing against.
acs_distances = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Submit all genomes for processing, where each genome will be processed in parallel
    futures = [executor.submit(calculate_acs_for_genome, genome) for genome in data["Sequence"]]

    # As each future completes, collect the results
    for future in concurrent.futures.as_completed(futures):
        acs_distances.append(future.result())

end = time.time()

print(f"Time taken: {end - start} seconds")


# In[9]:
acs_data = pd.DataFrame(acs_distances)
acs_data["Target"] = data["Lineage"].tolist()
acs_data["Test"] = data["Test"].tolist()
acs_data.to_parquet('../../data/features/acs.parquet', engine='pyarrow')