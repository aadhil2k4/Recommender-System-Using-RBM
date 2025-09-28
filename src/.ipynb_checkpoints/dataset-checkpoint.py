from torch.utils.data import Dataset
import numpy as np
import scipy.sparse as sp
import torch

from src.config import USERITEM_BINARY 

class UserItemDataset(Dataset):
    def __init__(self, ratings_npz_path=USERITEM_BINARY):  
        arr = sp.load_npz(str(ratings_npz_path)).tocoo()
        self.n_users, self.n_items = arr.shape
        # keep as csr for fast row access
        self.csr = arr.tocsr()

    def __len__(self):
        return self.n_users

    def __getitem__(self, idx):
        row = self.csr.getrow(idx).toarray().ravel()
        mask = (row != 0).astype(np.float32)
        # convert to torch
        return torch.from_numpy(row.astype(np.float32)), torch.from_numpy(mask)
