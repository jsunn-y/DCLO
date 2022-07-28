import numpy as np

index2base_dict = {
    0: "A",
    1: "C",
    2: "G",
    3: "T"
}

seq2encoding_dict = {
    "Z": (0, 0, 0, 0),
    "A": (1, 0, 0, 0),
    "C": (0, 1, 0, 0),
    "G": (0, 0, 1, 0),
    "T": (0, 0, 0, 1),
    "R": (1, 0, 1, 0),
    "Y": (0, 1, 0, 1),
    "M": (1, 1, 0, 0),
    "K": (0, 0, 1, 1),
    "S": (0, 1, 1, 0),
    "W": (1, 0, 0, 1),
    "H": (1, 1, 0, 1),
    "B": (0, 1, 1, 1),
    "V": (1, 1, 1, 0),
    "D": (1, 0, 1, 1),
    "N": (1, 1, 1, 1)
    }

encoding2choices_dict = {
    (1, 0, 0, 0): ('A',),
    (0, 1, 0, 0): ('C',),
    (0, 0, 1, 0): ('G',),
    (0, 0, 0, 1): ('T',),
    (1, 0, 1, 0): ('A', 'G'),
    (0, 1, 0, 1): ('C', 'T'),
    (1, 1, 0, 0): ('A', 'C'),
    (0, 0, 1, 1): ('G', 'T'),
    (0, 1, 1, 0): ('C', 'G'),
    (1, 0, 0, 1): ('A', 'T'),
    (1, 1, 0, 1): ('A', 'C', 'T'),
    (0, 1, 1, 1): ('C', 'G', 'T'),
    (1, 1, 1, 0): ('A', 'C', 'G'),
    (1, 0, 1, 1): ('A', 'G', 'T'),
    (1, 1, 1, 1): ('A', 'C', 'G', 'T')
    }

encoding2seq_dict = {v: k for k, v in seq2encoding_dict.items()}

def encoding2seq(encoding):
    encoding = encoding.reshape((-1, 4))
    seq = ""
    for row in encoding:
        seq += encoding2seq_dict[tuple(row)]
    return seq

def seq2encoding(seq):
    encoding = np.zeros((1, 4*len(seq)))
    for i, let in enumerate(seq):
        encoding[0, 4*i:4*(i+1)] = seq2encoding_dict[let]
    return encoding

def hammingdistance(seq1, seq2):
    return sum(s1 != s2 for s1, s2 in zip(seq1, seq2))