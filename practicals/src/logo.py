import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
from .seq_io import msa_2_num, AA_LIST, compute_freq
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

fp = FontProperties(family="sans-serif", weight="bold")
globscale = 1.35
AA_COL = {
    "A": "#80a0f0",  # Alanine
    "C": "#f08080",  # Cysteine
    "D": "#c04040",  # Aspartic acid
    "E": "#c04040",  # Glutamic acid
    "F": "#80c090",  # Phenylalanine
    "G": "#f09048",  # Glycine
    "H": "#4090d0",  # Histidine
    "I": "#80a0f0",  # Isoleucine
    "K": "#c040c0",  # Lysine
    "L": "#80a0f0",  # Leucine
    "M": "#80a0f0",  # Methionine
    "N": "#c04890",  # Asparagine
    "P": "#d0a030",  # Proline
    "Q": "#c04890",  # Glutamine
    "R": "#c040c0",  # Arginine
    "S": "#a0c048",  # Serine
    "T": "#a0c048",  # Threonine
    "V": "#80a0f0",  # Valine
    "W": "#80c090",  # Tryptophan
    "Y": "#60a0a0",  # Tyrosine
    "-": "#ffffff",  # Gap
    "X": "#ffffff",  # X
}
LETTERS = {aa: TextPath((-0.3, 0), aa, size=1, prop=fp) for aa in AA_COL}


def letterAt(letter, x, y, yscale=1, ax=None):
    text = LETTERS[letter]

    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    if yscale < 0:
        p = PathPatch(text, lw=0, fc=AA_COL[letter],  transform=t, alpha=0.7)
    else:
        p = PathPatch(text, lw=0, fc=AA_COL[letter],  transform=t)
    if ax != None:
        ax.add_artist(p)
    return p


def draw_logo(msa, ax, x=0, counts=False):
    if counts is False:
        positions = compute_freq(msa.values() if type(msa) is dict else msa)
    else:
        positions = msa
    maxi = 0
    for pos in positions:
        y = 0
        pos.sort(key=lambda el: el[1])
        for base, score in pos:
            letterAt(base, x, y, score, ax)
            y += score
        x += 1
        maxi = max(maxi, y)
    for axis in ['top','right']:
        ax.spines[axis].set_visible(False)
    ax.set_xlim([0, len(positions)])


def draw_msa(seqs, ax):
    if type(seqs) is list:
        num_msa = msa_2_num(seqs)
    elif type(seqs) is dict:
        num_msa = msa_2_num(seqs.values())
    boundaries = [i-0.5 for i in range(len(AA_LIST))]
    cmap = ListedColormap([AA_COL[aa] for aa in AA_LIST])
    ax.pcolormesh(num_msa, cmap=cmap)
    for axis in ['top','right', 'left', 'bottom']:
        ax.spines[axis].set_visible(False)
