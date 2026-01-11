import torch
import torch.nn as nn
import torch.nn.functional as F

class BDHProbe(nn.Module):
    def __init__(self, n_neurons=1024, d_rank=384):
        super().__init__()
        self.n = n_neurons
        self.d = d_rank

        self.decoder_x = nn.Parameter(torch.randn(n_neurons, d_rank))
        self.decoder_rho = nn.Parameter(torch.randn(d_rank, d_rank))

    def imprint_backstory(self, backstory_emb):
        return torch.tanh(self.decoder_rho @ backstory_emb.T).T

    def calculate_step(self, x_prev, rho_prev, v):
        x_new = x_prev + F.relu((self.decoder_x @ v.T).T)
        rho_new = rho_prev + 0.05 * torch.tanh(v)
        tension = torch.norm(x_new - x_prev, p=2)
        return x_new, rho_new, tension
