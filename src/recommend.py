import torch
import numpy as np
import pandas as pd

from src.config import RBM_CKPT, AE_CKPT, MOVIE_MAP, BINARY_NPZ, RATINGS_NPZ
from src.models.rbm import RBM
from src.models.autoencoder import Autoencoder

# Load movie mapping
movie_map = pd.read_csv(MOVIE_MAP)

def load_rbm(n_visible, n_hidden):
    model = RBM(n_visible=n_visible, n_hidden=n_hidden)
    model.load_state_dict(torch.load(RBM_CKPT, map_location="cpu"))
    model.eval()
    return model

def load_ae(n_items, hidden_dims, latent_dim):
    model = Autoencoder(n_items=n_items, hidden_dims=hidden_dims, latent_dim=latent_dim)
    model.load_state_dict(torch.load(AE_CKPT, map_location="cpu"))
    model.eval()
    return model

def recommend_for_user(user_id, model_type="rbm", top_k=10):
    if model_type == "rbm":
        import scipy.sparse as sp
        arr = sp.load_npz(BINARY_NPZ).tocsr()
        user_vector = arr[user_id].toarray().astype(np.float32)

        model = load_rbm(n_visible=arr.shape[1], n_hidden=256)
        v = torch.from_numpy(user_vector)
        with torch.no_grad():
            recon = model.sample_hidden_to_visible(model.sample_visible_to_hidden(v)).numpy().ravel()

    elif model_type == "ae":
        import scipy.sparse as sp
        arr = sp.load_npz(RATINGS_NPZ).tocsr()
        user_vector = arr[user_id].toarray().astype(np.float32)

        model = load_ae(n_items=arr.shape[1], hidden_dims=[1024,512], latent_dim=128)
        v = torch.from_numpy(user_vector)
        with torch.no_grad():
            recon = model(v).numpy().ravel()

    else:
        raise ValueError("Unknown model type, use 'rbm' or 'ae'")

    # Mask out movies the user has already rated
    seen = user_vector.ravel() > 0
    recon[seen] = -np.inf

    # Get top-K
    top_idx = np.argsort(recon)[::-1][:top_k]
    recommendations = movie_map.loc[movie_map["movieId"].isin(top_idx), ["movieId","title"]]
    return recommendations
