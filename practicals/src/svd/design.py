"""Design with SVD
"""
from .seq_io import encode_align, decode_seq
from .nuc import NUC, NUC_B, NUC_P
import numpy as np
from numpy import nan, cov, arange
from numpy import eye, diag
from numpy.linalg import svd
from numpy.random import multivariate_normal, normal, uniform, randint
import matplotlib.pyplot as plt


def sample_seq(seq_list:[str], nb:int=1, no_gap:bool=False, ref_seq:str=None,
               s_cov:str="cov", r:int=None, first:float=None)->[str]:
    """
    Keyword Arguments:
    seq_list:[str] -- List of sequences
    nb:int         -- (default 1) number of sequence to sample
    no_gap         -- (default True) whether to allow gap sampling
    Return:
    new_seq:[str]  -- List of sampled sequences
    """
    if ref_seq is not None:
        enc_seq = encode_align(seq_list) * (1 - encode_align([ref_seq]))
        mean_seq = enc_seq.mean(axis=0)
        u, s, vh = svd(enc_seq - mean_seq)
    else:
        enc_seq = encode_align(seq_list)
        mean_seq = enc_seq.mean(axis=0)
        u, s, vh = svd(enc_seq - mean_seq)

    if r is not None:
        u, s, vh = u[:, :r], s[:r], vh[:r, :]

    if first is not None:
        s[0] = first
    avg_u = u.mean(axis=0)
    var_u = u.var(axis=0)

    if s_cov == "cov":
        samp_seq = multivariate_normal(avg_u, cov(u, rowvar=False), size=nb)
    elif s_cov == "diag":
        samp_seq = multivariate_normal(avg_u, diag(var_u), size=nb)
    elif s_cov == "usamp":
        random_indices = randint(u.shape[0], size=(nb, u.shape[1]))
        # Use the indices to sample from each column of the matrix
        samp_seq = u[random_indices, arange(u.shape[1])]
    else:
        samp_seq = normal(size=(nb, u.shape[0]))

    if u.shape[0] > vh.shape[0]:
        new_bmsa = (samp_seq[:, :s.shape[0]] * s) @ vh
    else:
        new_bmsa = samp_seq @ (s * vh[:s.shape[0]].T).T
    new_bmsa = new_bmsa + mean_seq

    if no_gap:
        # do not sample gaps
        seq_len_5 = new_bmsa.shape[1]
        new_bmsa = new_bmsa.reshape(-1, seq_len_5//len(NUC), len(NUC))[:, :, :-1].reshape(-1, (len(NUC)-1)*(seq_len_5//len(NUC)))
    # plt.hist(new_bmsa.flatten())
    # plt.show()
    # if gaps are WT
    new_seq = decode_seq(new_bmsa, neg=no_gap and ref_seq is not None)
    return new_seq
