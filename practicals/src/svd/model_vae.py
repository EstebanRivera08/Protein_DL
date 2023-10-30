from torch import nn, zeros as tzeros
from torch import unsqueeze, exp as texp, reshape, log as tlog
import torch

torch.autograd.set_detect_anomaly(True)

class VAE_seq(nn.Module):

    def __init__(self, dim_node=5*10, nb_state=5, nb_hid=2, dim_hid=256,
                 dim_lat=10, device=None):
        super(VAE_seq, self).__init__()
        self.dim_lat = dim_lat
        self.dim_node = dim_node
        self.nb_pos = dim_node // 21
        self.nb_state = nb_state

        self.enc_node = nn.ModuleList([Block(dim_node, dim_hid)])
        for li in range(nb_hid-1):
            self.enc_node += [Block(dim_hid, dim_hid)]
        self.enc_node += [Block(dim_hid, dim_lat, act="id")]
        self.enc_si = Block(dim_hid, dim_lat, act="id")

        self.dec_node = nn.ModuleList([Block(dim_lat, dim_hid)])
        for li in range(nb_hid-1):
            self.dec_node += [Block(dim_hid, dim_hid)]
        self.dec_node += [Block(dim_hid, dim_node, act="id")]
        if device is not None:
            self.dev = device
        else:
            self.dev = torch.device("cpu")
        self.to(device)

    def encoder(self, conf):
        for lin in self.enc_node[:-1]:
            conf = lin(conf.to(self.dev))
        conf.to(self.dev)
        mu = self.enc_node[-1](conf.to(self.dev))
        sig = self.enc_si(conf.to(self.dev))
        eps = tzeros(mu.shape).data.normal_(0, 1.).to(self.dev)
        return mu, sig

    def decoder(self, conf):
        # decode
        for lin in self.dec_node[:-1]:
            conf = lin(conf.to(self.dev))
        conf = self.dec_node[-1](conf)

        fixed_shape = tuple(conf.shape[0:-1])
        conf = unsqueeze(conf, -1)
        conf = conf.view(fixed_shape + (-1, self.nb_state))

        log_p = nn.functional.log_softmax(conf, dim=-1)
        # prob = nn.functional.softmax(conf, dim=-1)
        log_p = log_p.view(fixed_shape + (-1,))
        # prob = prob.view(fixed_shape + (-1,))
        return log_p

    def forward(self, x):
        # encode
        mu, sig = self.encoder(x)
        eps = torch.randn_like(sig)
        z = mu + texp(sig) * eps

        log_p = self.decoder(z)
        kld = (0.5*(1 + sig - mu**2 - texp(sig))).sum(-1).to(self.dev)

        log_lk = (log_p.to(self.dev) * x.to(self.dev)).sum(-1)
        elbo = (log_lk + kld).mean()
        return elbo


class Block(nn.Module):

    def __init__(self, dim_in, dim_out, act="relu"):
        "Simple block"
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.Dropout(p=0.1),
            nn.ReLU() if act == "relu" else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)
