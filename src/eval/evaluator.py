import numpy as np
import torch
from scipy import sparse
from .metrics import precision_at_k, recall_at_k, ndcg_at_k
from ..models.autoencoder import Autoencoder
from ..models.rbm import RBM
from ..config import AE_CKPT, RBM_CKPT

def load_sparse_npz(path):
    return sparse.load_npz(str(path)).tocsr()

def evaluate_topk(model, user_vecs_csr, train_mask_csr, k=10, device="cpu", model_type="ae"):
    """
    model_type: "ae" or "rbm"
    user_vecs_csr: csr matrix of full rating matrix (for scoring)
    train_mask_csr: csr mask of training interactions (to exclude from ranking)
    """
    n_users, n_items = user_vecs_csr.shape
    precisions, recalls, ndcgs = [], [], []

    # convert training mask to lists for quick lookup
    for u in range(n_users):
        # target items are those in the test set (we'll derive outside ideally)
        # Here assume user_vecs_csr contains ground truth test ratings with nonzero entries.
        true_items = set(user_vecs_csr.getrow(u).nonzero()[1])
        if len(true_items) == 0:
            continue
        # build input: training vector (zeros where training missing)
        train_vec = train_mask_csr.getrow(u).toarray().ravel()
        # score items
        if model_type == "ae":
            model.eval()
            with torch.no_grad():
                x = torch.tensor(train_vec).float().unsqueeze(0).to(device)
                scores = model(x).cpu().numpy().ravel()
        else:
            # RBM: use hidden->visible reconstruction
            model.eval()
            with torch.no_grad():
                v = torch.tensor(train_vec).float().unsqueeze(0).to(device)
                scores = model(v).cpu().numpy().ravel()

        # mask out training items so ranking only on unseen
        train_items = set(train_mask_csr.getrow(u).nonzero()[1])
        for itm in train_items:
            scores[itm] = -np.inf

        precisions.append(precision_at_k(scores, true_items, k=k))
        recalls.append(recall_at_k(scores, true_items, k=k))
        ndcgs.append(ndcg_at_k(scores, true_items, k=k))

    return {
        "precision@k": np.mean(precisions) if precisions else 0.0,
        "recall@k": np.mean(recalls) if recalls else 0.0,
        "ndcg@k": np.mean(ndcgs) if ndcgs else 0.0,
    }
