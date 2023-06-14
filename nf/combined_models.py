import torch
import torch.nn as nn
import numpy as np
from nf.cnn_3d_stack import CNN3D_stackout
if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """

    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, x):
        return self.network(x)


class COMBINED_Model(nn.Module):
    """
    Combined model for the AR_NPE.
    """

    def __init__(
        self,
        # priors_all,
        Mdiff_model,
        # M1_model,
        # Ntot_model,
        ndim,
        ksize,
        nside_in,
        nside_out,
        nbatch,
        ninp,
        nfeature,
        nout,
        layers_types=['cnn', 'res', 'res', 'res'],
        act='tanh',
        padding='valid',
        # sep_Ntot_cond=False,
        # sep_M1_cond=False,
        sep_Mdiff_cond=False,
        ):
        super().__init__()
        # self.priors_all = priors_all
        # self.M1_model = M1_model
        # self.Ntot_model = Ntot_model
        self.Mdiff_model = Mdiff_model
        self.nbatch = nbatch
        self.nout = nout
        self.ninp = ninp

        self.conv_layers = CNN3D_stackout(
            ksize,
            nside_in,
            nside_out,
            nbatch,
            ninp,
            nfeature,
            nout,
            layers_types=layers_types,
            act=act,
            padding=padding
            )
        self.ndim = ndim
        self.sep_Mdiff_cond = sep_Mdiff_cond
        if self.sep_Mdiff_cond:
            self.cond_Mdiff_layer = FCNN(nout, nout, nout)

    def forward(
        self,
        x_Mdiff,
        cond_x=None,
        mask_Mdiff_truth_all=None
        ):
        nbatches = cond_x.shape[0]
        logP_Ntot = torch.zeros(1, device=device)
        logP_M1 = torch.zeros(1, device=device)
        logP_Mdiff = torch.zeros(1, device=device)
        for jb in range(nbatches):
            cond_out = self.conv_layers(cond_x[jb])
            mask_Mdiff_truth = mask_Mdiff_truth_all[jb].to(device)
            if self.sep_Mdiff_cond:
                cond_inp_Mdiff = self.cond_Mdiff_layer(cond_out)
            if jb == 0:
                logP_Mdiff = self.Mdiff_model.forward(x_Mdiff[jb], cond_inp_Mdiff, mask_Mdiff_truth)
            else:
                logP_Mdiff += self.Mdiff_model.forward(x_Mdiff[jb], cond_inp_Mdiff, mask_Mdiff_truth)
        loss = torch.mean(-logP_Mdiff)

        return loss

    def inverse(
        self,
        cond_x=None,
        mask_Mdiff_truth=None,
        ):
        nbatches = cond_x.shape[0]
        M_diff_samp_out = []
        for jb in range(nbatches):
            cond_out = self.conv_layers(cond_x[jb])
            if self.sep_Mdiff_cond:
                cond_out = self.cond_Mdiff_layer(cond_out)
            M_diff_samp, _ = self.Mdiff_model.inverse(cond_out, mask_Mdiff_truth)
            M_diff_samp_out.append(M_diff_samp)
        return M_diff_samp_out
