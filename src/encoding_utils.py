index2base_dict = {
    0: "A",
    1: "C",
    2: "G",
    3: "T"
}

seq2encoding_dict = {
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
    "N": (1, 1, 1, 1),
    }

encoding2seq_dict = {v: k for k, v in seq2encoding_dict.items()}

def encoding2seq(encoding):
    encoding = encoding.reshape((-1, 4))
    seq = ""
    for row in encoding:
        seq += encoding2seq_dict[tuple(row)]
    return seq