import torch
import numpy as np
from pathlib import Path
from ..models.autoencoder import Autoencoder
from ..models.rbm import RBM
from ..utils.io import load_model_state
import joblib

def load_ae(path, n_items, hidden_dims, latent_dim, device="cpu"):
    model = Autoencoder(n_items=n_items, hidden_dims=hidden_dims, latent_dim=latent_dim)
    return load_model_state(model, Path(path), device=device)

def load_rbm(path, n_visible, n_hidden, k=1, device="cpu"):
    model = RBM(n_visible=n_visible, n_hidden=n_hidden, k=k)
    return load_model_state(model, Path(path), device=device)

def recommend_ae(model, user_vector, train_mask, top_k=10, device="cpu"):
    # user_vector: numpy vector shaped (n_items,) with zeros for missing
    model.eval()
    with torch.no_grad():
        x = torch.tensor(user_vector).float().unsqueeze(0).to(device)
        scores = model(x).cpu().numpy().ravel()
    scores[train_mask.nonzero()] = -np.inf
    idx = np.argsort(-scores)[:top_k]
    return idx, scores[idx]

def recommend_rbm(model, user_vector, train_mask, top_k=10, device="cpu"):
    model.eval()
    with torch.no_grad():
        v = torch.tensor(user_vector).float().unsqueeze(0).to(device)
        out = model(v).cpu().numpy().ravel()
    out[train_mask.nonzero()] = -np.inf
    idx = np.argsort(-out)[:top_k]
    return idx, out[idx]
