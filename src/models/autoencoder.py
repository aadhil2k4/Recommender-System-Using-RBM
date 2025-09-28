import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, n_items, hidden_dims=[1024,512], latent_dim=128, dropout=0.3):
        super().__init__()
        dims = [n_items] + hidden_dims + [latent_dim]
        enc_layers = []
        for i in range(len(dims)-1):
            enc_layers.append(nn.Linear(dims[i], dims[i+1]))
            enc_layers.append(nn.ReLU())
            if i < len(hidden_dims):
                enc_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*enc_layers)

        # decoder mirror
        dec_dims = [latent_dim] + hidden_dims[::-1] + [n_items]
        dec_layers = []
        for i in range(len(dec_dims)-1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i+1]))
            if i < len(dec_dims)-2:
                dec_layers.append(nn.ReLU())
                dec_layers.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

def masked_mse_loss(pred, target, mask):
    
    diff = (pred - target) * mask
    num_obs = mask.sum()
    if num_obs == 0:
        return torch.tensor(0.0, device=pred.device)
    return torch.sum(diff ** 2) / num_obs
