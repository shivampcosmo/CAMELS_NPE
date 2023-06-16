import torch
import torch.nn as nn
import numpy as np
from nf.cnn_3d_stack import CNN3D_stackout
if torch.cuda.is_available():    
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")    
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
        Reg_model,
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
        sep_Reg_cond=False,
        ):
        super().__init__()
        self.Reg_model = Reg_model
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
        self.sep_Reg_cond = sep_Reg_cond
        if self.sep_Reg_cond:
            self.cond_Reg_layer = FCNN(nout, nout, nout)

    def forward(
        self,
        x_Reg,
        cond_x=None,
        mask_Reg_truth_all=None
        ):
        nbatches = cond_x.shape[0]
        logP_Ntot = torch.zeros(1, device=device)
        logP_M1 = torch.zeros(1, device=device)
        logP_Reg = torch.zeros(1, device=device)
        for jb in range(nbatches):
            cond_out = self.conv_layers(cond_x[jb])
            mask_Reg_truth = mask_Reg_truth_all[jb].to(device)
            if self.sep_Reg_cond:
                cond_inp_Reg = self.cond_Reg_layer(cond_out)
            if jb == 0:
                logP_Reg = self.Reg_model.forward(x_Reg[jb], cond_inp_Reg, mask_Reg_truth)
            else:
                logP_Reg += self.Reg_model.forward(x_Reg[jb], cond_inp_Reg, mask_Reg_truth)
        loss = torch.mean(-logP_Reg)

        return loss

    def inverse(
        self,
        cond_x=None,
        mask_Reg_truth=None,
        ):
        nbatches = cond_x.shape[0]
        M_diff_samp_out = []
        for jb in range(nbatches):
            cond_out = self.conv_layers(cond_x[jb])
            if self.sep_Reg_cond:
                cond_out = self.cond_Reg_layer(cond_out)
            M_diff_samp, _ = self.Reg_model.inverse(cond_out, mask_Reg_truth)
            M_diff_samp_out.append(M_diff_samp)
        return M_diff_samp_out
