from numpy import array, zeros, diag, sign, argmax, zeros_like, copy
from numpy.linalg import svd
from .seq_io import trim_msa, encode_align
from .nuc import NUC, NUC_B, NUC_P
import matplotlib.pyplot as plt


def reweight_msa(bmsa, cor_thres=0.8):
    cor = bmsa @ bmsa.T/(bmsa.shape[1]//len(NUC))
    iwei = array([sum(cor[p, :] > cor_thres) for p in range(bmsa.shape[0])])
    return bmsa/iwei.reshape(bmsa.shape[0], 1)


def plot_pca(pi, pj, pca_f, col=None):
    delta_x = (abs(pi.min()-pi.max()))*0.05
    delta_y = (abs(pj.min()-pj.max()))*0.05
    pca_f.plot([0, 0], [pj.min()-delta_y, pj.max()+delta_y], linestyle="--", alpha=0.5, c="grey", linewidth=1)
    pca_f.plot([pi.min()-delta_x, pi.max()+delta_x], [0, 0], linestyle="--", alpha=0.5, c="grey", linewidth=1)
    pca_f.scatter(pi, pj, s=3, c="lightsteelblue" if col is None else col, marker="o",)
    pca_f.set_xticks([])
    pca_f.set_yticks([])
    pca_f.set_xlim([pi.min()-delta_x, pi.max()+delta_x])
    pca_f.set_ylim([pj.min()-delta_y, pj.max()+delta_y])
    for axis in ['top', 'right']:
        pca_f.spines[axis].set_visible(False)
    pca_f.set_xlabel(f"PC")
    pca_f.set_ylabel(f"PC")
    pca_f.set_aspect(1)


def SVD(sequences, ref_seq=None, trim=False, bmsa=False, reweight=False,
        center_ref=True, no_centering=False):
    "sequences = dictionary {name: seq}"
    if trim and ref_seq is not None:
        sequences, ref_seq = trim_msa(sequences, ref_seq)

    enc_seq = encode_align(list(sequences.values()) if type(sequences) is dict else sequences)

    if reweight:
        enc_seq = reweight_msa(enc_seq)

    if ref_seq is None:
        if not no_centering:
            enc_seq = enc_seq - enc_seq.mean(axis=0)
    else:
        enc_ref = encode_align([ref_seq])

        if not no_centering:
            if center_ref:
                enc_seq = enc_seq * (1 - enc_ref)
            else:
                enc_seq = enc_seq - enc_seq.mean(axis=0)

    u, s, vh = svd(enc_seq, full_matrices=True)

    # flip all the vectors
    pca_dir = copy(vh.T)
    min_r = min(vh.shape[0], u.shape[0])
    pca_dir[:, :min_r] = sign(u.sum(axis=0))[:min_r].reshape(1, -1) * pca_dir[:, :min_r]
    svd_out = (u, s, vh)
    if bmsa:
        return u * s, svd_out, pca_dir, enc_seq
    else:
        return u * s, svd_out, pca_dir
