"""
The Ambiguous Codes
Y :: C or T
R :: A or G
M :: A or C
K :: G or T
S :: C or G
W :: A or T
B :: C or G or T
D :: A or G or T
H :: A or C or T
V :: A or C or G
N :: A or C or G or T
"""

from collections import Counter

def ambiguous_bases_count(seq):
    bases = ["Y", "R", "K", "S", "W", "M", "B", "D", "H", "V", "N"]
    counts = Counter(seq)
    total_count = sum(counts[base] for base in bases)
    return total_count

def regular_bases_count(seq):
    bases = ["A", "C", "G", "T"]
    counts = Counter(seq)
    total_count = sum(counts[base] for base in bases)
    return total_count

def ambiguity_score(sequence: str):
    ambiguous_codes = ['Y', 'R', 'M', 'K', 'S', 'W', 'B', 'D', 'H', 'V', 'N']
    code_weights = {
        'Y': 2,
        'R': 2,
        'M': 2,
        'K': 2,
        'S': 2,
        'W': 2,
        'B': 3,
        'D': 3,
        'H': 3,
        'V': 3,
        'N': 4
    }
    counter = Counter(sequence)
    score = sum(counter[code] * code_weights[code] for code in ambiguous_codes)
    total_weights = sum(code_weights[code] for code in code_weights)

    return score / total_weights
