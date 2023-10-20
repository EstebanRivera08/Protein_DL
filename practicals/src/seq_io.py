"""Read files
"""
from numpy import array
from collections import Counter

AA_LIST = ["A" , "C" , "T" , "E" , "D" , "F" , "W" , "I" , "V" , "L" , "K" , "M" , "N" , "Q" , "S" , "R" , "Y" , "H" , "P" , "G", "-"]
AA_DIC = {"ALA": "A" , "CYS": "C" , "THR": "T" , "GLU": "E" , "ASP": "D" , "PHE": "F" , "TRP": "W" , "ILE": "I" , "VAL": "V" , "LEU": "L" , "LYS": "K" , "MET": "M" , "ASN": "N" , "GLN": "Q" , "SER": "S" , "ARG": "R" , "TYR": "Y" , "HIS": "H" , "PRO": "P" , "GLY": "G"}
RAA_DIC = {v: k for k, v in AA_DIC.items()}


def msa_2_num(msa):
    num_msa = []
    for seq in msa:
        n_seq = [AA_LIST.index(aa if aa in AA_LIST else "-")  for aa in seq]
        num_msa += [n_seq]
    return array(num_msa)


def compute_freq(msa):
    num_msa = msa_2_num(msa)
    count_msa = []
    for pos in range(num_msa.shape[1]):
        count_msa += [[(AA_LIST[aa], c/num_msa.shape[0]) for aa, c in Counter(num_msa[:, pos]).items()]]
    return count_msa


def write_fasta(name, seq, outfile):
    "write fasta"
    results = {}
    with open(outfile, "w") as out:
        out.write(f">{name}\n{seq}\n")


def read_fasta(infile):
    results = {}
    for l in open(infile):
        if l.startswith(">"):
            name = l.strip()[1:]
            results[name] = ""
        else:
            results[name] += l.strip()
    return results


def read_seq(infile):
    results = {}
    for si, l in enumerate(open(infile)):
        results[si] = l.strip()
    return results


def trim_msa(msa, ref_seq):
    pos = [i for i, si in enumerate(ref_seq) if si != "-"]
    if type(msa) is dict:
        new_msa = {}
        for n, seq in msa.items():
            new_msa[n] = "".join([seq[i] for i in pos])
    else:
        new_msa = []
        for seq in msa:
            new_msa += ["".join([seq[i] for i in pos])]
    return new_msa, "".join([ref_seq[i] for i in pos])
