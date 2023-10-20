"""Read files
"""

from numpy import triu_indices_from, diag_indices_from, array, argmax, identity, ndarray, int16, zeros, einsum
from .nuc import NUC, NUC_B, NUC_P


def num_to_seq(num_seq):
    "convert num to seq"
    return "".join([NUC[el] for el in num_seq])


def seq_to_index(seq):
    res = []
    for el in seq:
        res += [NUC.index(el) if el in NUC else NUC.index("-")]
    return res


def msa_to_index(msa):
    res_msa = []
    for seq in msa:
        res_msa += [seq_to_index(seq)]
    return array(res_msa)


def encode_seq(seq, neg=False):
    res = []
    for el in seq:
        if neg:
            res += NNUC_B[NUC.index(el)] if el in NUC else NNUC_B[NUC.index("-")]
        else:
            res += NUC_B[NUC.index(el)] if el in NUC else NUC_B[NUC.index("-")]
    return res


def decode_seq(bmsa, neg=False):
    msa = []
    bsize, seq_len_5 = bmsa.shape
    if neg:
        bmsa = bmsa.reshape(-1, seq_len_5//len(NUC)-1, len(NUC)-1)
    else:
        bmsa = bmsa.reshape(-1, seq_len_5//len(NUC), len(NUC))
    for batch in bmsa:
        msa += ["".join([NUC[n] for n in argmax(batch, axis=-1)])]
    return msa


def decode_index(bmsa, neg=False):
    msa = []
    bsize, seq_len_5 = bmsa.shape
    for batch in bmsa:
        msa += ["".join([NUC[n] for n in batch])]
    return msa


def encode_align(msa, neg=False):
    if type(msa) is dict:
        msa = msa.values()
    res_msa = []
    for seq in msa:
        res_msa += [encode_seq(seq, neg)]
    return array(res_msa)


def pair_frequency(msa, no_gap=False, batch=10, ein_f=False):
    """Compute the pairwise frequencies
    """
    # bmsa = encode_align(msa).astype(int16)
    bmsa = encode_align(msa)
    nb_seq, len5 = bmsa.shape
    if no_gap:
        assert len5 % len(NUC) == 0, "error format binary MSA"
        bmsa = bmsa.reshape(nb_seq, len5//len(NUC), len(NUC))[:, :, :-1].reshape(nb_seq, (len(NUC)-1)*(len5//len(NUC)))
    nb_seq, seq_sp = bmsa.shape
    if not ein_f:
        res_mat = zeros((seq_sp, seq_sp))
        num_chunks = (nb_seq + batch - 1) // batch
        for i in range(num_chunks):
            start = i * batch
            end = min((i + 1) * batch, nb_seq)
            chunk = bmsa[start:end]
            reshaped = chunk.reshape(-1, 1, seq_sp)
            res_chunk = (reshaped * reshaped.transpose((0, 2, 1))).sum(axis=0)
            res_mat += res_chunk
    else:
        res_mat = einsum('ij,kl->ijl', bmsa, bmsa).mean(axis=0)
    res_mat[diag_indices_from(res_mat)] = 0
    return res_mat.flatten()/nb_seq


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


def get_chimeric(msa, ref_seq):
    """
    Replace gaps with the reference sequence nucleotides
    """
    if type(msa) == list:
        new_msa = []
        for seq in msa:
            nseq = "".join([n if n != "-" else r for n, r in zip(seq, ref_seq)])
            new_msa += [nseq]
    elif type(msa) == dict:
        new_msa = {}
        for name, seq in msa.items:
            nseq = "".join([n if n != "-" else r for n, r in zip(seq, ref_seq)])
            new_msa[name] = nseq
    return new_msa
