"""Train VAE to reproduce sequence frequencies
"""

import argparse
from torch import save as tsave, load as tload, optim, no_grad, log as tlog, diag
from torch import set_num_threads, cuda, device as tdevice, argmax, randn, float as tfloat
from torch import eye as teye, cat as tcat, zeros as tzeros, no_grad, tensor, randn_like

from torch.autograd import set_detect_anomaly
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal

from .model_vae import VAE_seq
from .seq_io import encode_align, decode_index

from numpy import zeros, array, sum as npsum, exp, eye as teye, diag, mean
from numpy.random import uniform

from torch.utils.data import Dataset, DataLoader

class MSA(Dataset):

    def __init__(self, seq_list):
        bmsa = encode_align(seq_list)
        self.data = tensor(bmsa, dtype=tfloat)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def init_model(len_seq, device=None):
    """Default model parameters
    """
    nb_state = 21
    dim_node = len_seq * nb_state
    nb_hid = 1
    dim_hid = 2**9
    dim_lat = 2**7
    model = VAE_seq(dim_node, nb_state=nb_state, nb_hid=nb_hid, dim_hid=dim_hid,
                    dim_lat=dim_lat, device=device)
    return model


def train_model(seq_list, model, bsize=100, lr=10**-3, nb_iter=100, init_parms=None):
    if init_parms is not None:
        model.load_state_dict(tload(init_parms, map_location=model.dev))

    out_file = "parms/parms_{}.dat"
    model.train()
    # get the data
    optimizer = optim.Adam(model.parameters(), weight_decay=0.01)
    bin_msa = MSA(seq_list)

    loss_epoch = []
    for epoch in range(nb_iter):
        loader = DataLoader(bin_msa, batch_size=bsize, shuffle=True)
        loss_batch = []
        for bi, batch in enumerate(loader):
            optimizer.zero_grad()
            loss = -model(batch)
            loss.backward()
            optimizer.step()
            loss_batch += [loss.item()]
        loss_epoch += [mean(loss_batch)]
    return loss_epoch


def generate_seq(model, nb_seq, no_gap=False, scale_var=1.):
    "generate random sequences"
    z_ls = randn(nb_seq, model.dim_lat)

    seq = model.decoder(z_ls.float())
    seq = seq.reshape(nb_seq, model.nb_pos, 21)

    if no_gap:
        max_p = argmax(seq[:, :, :-1], -1)
    else:
        max_p = argmax(seq, -1)

    return decode_index(max_p)
