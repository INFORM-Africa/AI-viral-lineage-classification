"""
This is the implementation of the feature extraction technique proposed in 
Murugaiah, Muthulakshmi & Ganesan, Murugeswari. (2021). A Novel Frequency Based 
Feature Extraction Technique for Classification of Corona Virus Genome and Discovery
of COVID-19 Repeat Pattern. Brazilian Archives of Biology and Technology. 64. 
doi : http://dx.doi.org/10.1590/1678-4324-2021210075
"""
from collections import Counter
from typing import Iterable, Tuple
import itertools
import math
import numpy as np
from timebudget import timebudget

class MurugaiahFeatures:
    def __init__(self):
        pass
    
    def extract(self, sequences:Iterable[str], fourier:bool=False) -> np.ndarray:
        features = [self.extract_single(seq=sequence, f=fourier) for sequence in sequences]
        return np.array(features)
    
    def extract_single(self, seq:str, f:bool=False) -> np.array:
        f1_features = self._feature_based_on_storage(seq) # 2 features
        f2_features = self._feature_based_on_bases_frequency(seq) # 85 features
        f4_features = self._feature_based_on_composition_of_amino_acids(seq) # 27 features
        f3_features = self._feature_based_on_arrangement_of_patterns(seq) # 6 features
        features = np.concatenate((f1_features, f2_features, f3_features, f4_features)) # 120 features

        if f:
            features = np.fft.fft(features, axis=-1)
            
        return features

    @staticmethod
    def _feature_based_on_storage(sequence:str) -> Tuple[float, ...]:
        length = len(sequence)
        size = length/1024
        f1 = (length, size)
        return f1 # 2 features

    @staticmethod
    def _feature_based_on_bases_frequency(sequence:str) -> Tuple[float, ...]:
        bases = ['A', 'C', 'G', 'T']

        # N Count
        n_count = sequence.count('N') # 1 feature

        # Base count
        bases_occurrences_dict = Counter(sequence)
        bases_occurrences = [bases_occurrences_dict[base] for base in bases] # 4 features

        # Dimer Count
        dimers = [''.join(pair) for pair in itertools.product(bases, repeat=2)] 
        dimers_occurrences_dict = Counter(sequence[i:i + 2] for i in range(len(sequence) - 1))
        dimers_occurrences = [dimers_occurrences_dict[dimer] for dimer in dimers] # 16 features

        # Codon Count
        codons = [''.join(triplet) for triplet in itertools.product(bases, repeat=3)]
        codons_occurrences_dict = Counter(sequence[i:i + 3] for i in range(len(sequence) - 2))
        codons_occurrences = [codons_occurrences_dict[codon] for codon in codons] # 64 features

        f2 = (n_count, *bases_occurrences, *dimers_occurrences, *codons_occurrences)
        return f2 # 85 features
    
    @staticmethod
    def count_palindrome_patterns(sequence:str, max_length:int) -> int:
        length = len(sequence)
        dp = [[0 for x in range(length)] for y in range(length)]
        
        # All substrings of length 1 are palindromes
        for i in range(length):
            dp[i][i] = 1
        
        # Check for sub-string of length 2
        for i in range(length - 1):
            if sequence[i] == sequence[i+1]:
                dp[i][i+1] = 1
        
        # Check for lengths greater than 2
        for cl in range(3, max_length):
            for i in range(length - cl + 1):
                j = i + cl - 1
                # Check if the first and last characters are the same
                if (sequence[i] == sequence[j] and dp[i + 1][j - 1]):
                    dp[i][j] = 1
        
        # Return the count of palindrome patterns
        return sum(map(sum, dp))

    @staticmethod
    def _feature_based_on_arrangement_of_patterns(sequence:str, pal_limit=4) -> Tuple[float, ...]:
        sequence_length = len(sequence)

        # Most Repeat Pattern Count (MRP)
        tetranucleotides_occurrences_dict = Counter(sequence[i:i + 4] for i in range(len(sequence) - 3))
        most_repeated_count = max(tetranucleotides_occurrences_dict.values()) # 1 feature

        # Palindrome Count (PC) & Palindrome Threshold (PT)
        #palindrome_count = MurugaiahFeatures.count_palindrome_patterns(sequence, pal_limit) # 1 feature

        # palindrome_count = 0 # 1 feature
        # palindrome_threshold = 0 # 1 feature
        # if limit is None:
        #     limit = sequence_length
        
        # for start in range(sequence_length):
        #     for end in range(start + 1, sequence_length):
        #         subsequence = sequence[start:end]
        #         if subsequence == subsequence[::-1]:
        #             palindrome_count += 1
        #             palindrome_threshold = max(palindrome_threshold, len(subsequence))

        # Entropy
        bases_occurrences_dict = Counter(sequence)
        entropy = 0 # 1 feature
        for count in bases_occurrences_dict.values():
            if count > 0:
                probability = count / sequence_length
                entropy -= probability * math.log2(probability)

        # Open Reading Frame Count (ORF) & Open Reading Frame Threshold
        start_codon = "ATG"
        stop_codons = ["TAA", "TAG", "TGA"]

        orf_count = 0 # 1 feature
        orf_threshold = 0 # 1 feature
        
        for start in range(sequence_length - 2):
            if sequence[start:start+3] == start_codon:
                end = start + 3
                while end < sequence_length - 2:
                    if sequence[end:end+3] in stop_codons:
                        orf_count += 1
                        orf_threshold = max(orf_threshold, end - start + 3)
                        break
                    end += 3

        f3 = (most_repeated_count, entropy, orf_count, orf_threshold)
        # f3 = (most_repeated_count, palindrome_count, palindrome_threshold, entropy, orf_count, orf_threshold)
        return f3 # 6 features

    @staticmethod
    def _feature_based_on_composition_of_amino_acids(sequence:str) -> Tuple[float, ...]:
        # Amino acid Count
        codon_to_amino_acid = {
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
            'AAT': 'N', 'AAC': 'N',
            'GAT': 'D', 'GAC': 'D',
            'TGT': 'C', 'TGC': 'C',
            'CAA': 'Q', 'CAG': 'Q',
            'GAA': 'E', 'GAG': 'E',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
            'CAT': 'H', 'CAC': 'H',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I',
            'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'AAA': 'K', 'AAG': 'K',
            'ATG': 'M',
            'TTT': 'F', 'TTC': 'F',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'TGG': 'W',
            'TAT': 'Y', 'TAC': 'Y',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V'
        }
        amino_acid_counts = {aa: 0 for aa in set(codon_to_amino_acid.values())} # 20 features
        for i in range(0, len(sequence), 3):
            codon = sequence[i:i+3]
            if codon in codon_to_amino_acid:
                amino_acid = codon_to_amino_acid[codon]
                amino_acid_counts[amino_acid] += 1

        # Stop Codon Count
        stop_codons = ['TAA', 'TAG', 'TGA']
        stop_codon_count = 0 # 1 feature
        for i in range(len(sequence) - 2):
            if sequence[i:i+3] in stop_codons:
                stop_codon_count += 1

        # Atomic Composition
        atomic_composition = {
            'C': 0,  # Carbon
            'H': 0,  # Hydrogen (excluding terminal bond)
            'O': 0,  # Oxygen (excluding terminal bond)
            'N': 0,  # Nitrogen
            'S': 0   # Sulphur
        } # 5 features

        # Molecular formula of each amino acid from the previous table
        molecular_formula = {
                'A': {'C': 3, 'H': 7, 'O': 2, 'N': 1, 'S': 0},
                'R': {'C': 6, 'H': 14, 'O': 2, 'N': 4, 'S': 0},
                'N': {'C': 4, 'H': 8, 'O': 3, 'N': 2, 'S': 0},
                'D': {'C': 4, 'H': 7, 'O': 4, 'N': 0, 'S': 0},
                'C': {'C': 3, 'H': 7, 'O': 2, 'N': 1, 'S': 1},
                'Q': {'C': 5, 'H': 10, 'O': 3, 'N': 2, 'S': 0},
                'E': {'C': 5, 'H': 9, 'O': 4, 'N': 0, 'S': 0},
                'G': {'C': 2, 'H': 5, 'O': 2, 'N': 1, 'S': 0},
                'H': {'C': 6, 'H': 9, 'O': 2, 'N': 3, 'S': 0},
                'I': {'C': 6, 'H': 13, 'O': 2, 'N': 1, 'S': 0},
                'L': {'C': 6, 'H': 13, 'O': 2, 'N': 1, 'S': 0},
                'K': {'C': 6, 'H': 14, 'O': 2, 'N': 2, 'S': 0},
                'M': {'C': 5, 'H': 11, 'O': 2, 'N': 1, 'S': 1},
                'F': {'C': 9, 'H': 11, 'O': 2, 'N': 1, 'S': 0},
                'P': {'C': 5, 'H': 9, 'O': 2, 'N': 1, 'S': 0},
                'S': {'C': 3, 'H': 7, 'O': 3, 'N': 1, 'S': 0},
                'T': {'C': 4, 'H': 9, 'O': 3, 'N': 1, 'S': 0},
                'W': {'C': 11, 'H': 12, 'O': 2, 'N': 2, 'S': 0},
                'Y': {'C': 9, 'H': 11, 'O': 3, 'N': 1, 'S': 0},
                'V': {'C': 5, 'H': 11, 'O': 2, 'N': 1, 'S': 0}
        }
        terminal_H = 2
        terminal_O = 1
        
        for aa, count in amino_acid_counts.items():
            atomic_composition['H'] += (count * molecular_formula[aa]['H']) - terminal_H
            atomic_composition['O'] += (count * molecular_formula[aa]['O']) - terminal_O
            atomic_composition['C'] += count * molecular_formula[aa]['C']
            atomic_composition['N'] += count * molecular_formula[aa]['N']
            atomic_composition['S'] += count * molecular_formula[aa]['S']
                
        # Molecular Weight
        molar_masses = {
            'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.15,
            'Q': 146.15, 'E': 147.13, 'G': 75.07, 'H': 155.16, 'I': 131.17,
            'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
            'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15
        }
        water_weight = 18.015
        
        molecular_weight = sum(amino_acid_counts[aa] * molar_masses[aa] for aa in amino_acid_counts.keys()) 
        molecular_weight = molecular_weight - water_weight # 1 feature

        f4 = (*amino_acid_counts.values(), stop_codon_count, *atomic_composition.values(), molecular_weight)
        return f4 # 27 features
