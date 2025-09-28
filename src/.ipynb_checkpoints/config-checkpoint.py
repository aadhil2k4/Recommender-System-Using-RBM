from pathlib import Path
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RAW_DIR = os.path.join(ROOT,"data","raw","ml-1m")
PROCESSED_DIR = os.path.join(ROOT,"data","processed")

RATINGS_NPZ = os.path.join(PROCESSED_DIR,"userItem_ratings.npz")
BINARY_NPZ = os.path.join(PROCESSED_DIR,"userItem_binary.npz") 
TRAIN_CSV = os.path.join(PROCESSED_DIR,"train_ratings.csv")
TEST_CSV = os.path.join(PROCESSED_DIR,"test_ratings.csv")
MOVIE_MAP = os.path.join(PROCESSED_DIR,"movie_mapping.csv")

MODELS_DIR = os.path.join(ROOT,"model")

RBM_CKPT = os.path.join(MODELS_DIR,"rbm_model.pt")
AE_CKPT = os.path.join(MODELS_DIR,"autoencoder_model.pt")

USERITEM_BINARY = BINARY_NPZ  

DEFAULTS = {
    "rbm": {
        "n_hidden": 256,
        "lr": 1e-3,
        "batch_size": 128,
        "epochs": 30,
        "cd_k": 1,
        "weight_decay": 1e-4,
    },
    "ae": {
        "latent_dim": 128,
        "hidden_dims": [1024, 512],
        "lr": 1e-3,
        "batch_size": 128,
        "epochs": 30,
        "dropout": 0.3,
    },
}
