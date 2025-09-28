import torch
import torch.nn as nn

class RBM(nn.Module):
    
    def __init__(self, n_visible: int, n_hidden: int = 256, k: int = 1):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k

        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))

    def sample_h(self, v_prob):
        # v_prob: (batch, n_visible)
        p_h = torch.sigmoid(torch.matmul(v_prob, self.W.t()) + self.h_bias)
        return p_h, torch.bernoulli(p_h)

    def sample_v(self, h_prob):
        p_v = torch.sigmoid(torch.matmul(h_prob, self.W) + self.v_bias)
        return p_v, torch.bernoulli(p_v)

    def gibbs(self, v0):
        vk = v0
        for _ in range(self.k):
            ph, hk = self.sample_h(vk)
            pv, vk = self.sample_v(hk)
        return vk.detach()

    def forward(self, v):
        # return reconstruction probability
        ph, _ = self.sample_h(v)
        pv, _ = self.sample_v(ph)
        return pv

    def contrastive_divergence_update(self, v0, lr=1e-3, weight_decay=0.0):
        # v0: batch_size x n_visible (0/1)
        ph0, h0 = self.sample_h(v0)
        # positive gradient
        pos_grad = torch.matmul(ph0.t(), v0)

        # Gibbs chain
        vk = v0
        for _ in range(self.k):
            ph, hk = self.sample_h(vk)
            pv, vk = self.sample_v(hk)

        phk, _ = self.sample_h(vk)
        neg_grad = torch.matmul(phk.t(), vk)

        # parameter updates (manual)
        batch_size = v0.size(0)
        dW = (pos_grad - neg_grad) / batch_size - weight_decay * self.W
        dvb = torch.sum(v0 - vk, dim=0) / batch_size
        dhb = torch.sum(ph0 - phk, dim=0) / batch_size

        # apply updates in-place
        self.W.data += lr * dW
        self.v_bias.data += lr * dvb
        self.h_bias.data += lr * dhb

        # compute reconstruction loss 
        loss = torch.mean((v0 - vk) ** 2)
        return loss
