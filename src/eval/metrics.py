import numpy as np

def precision_at_k(pred_scores, true_items, k=10):
    """
    pred_scores: array of item scores length M
    true_items: set/list of item indices that are relevant (test)
    """
    idx = np.argsort(-pred_scores)[:k]
    hits = sum(1 for i in idx if i in true_items)
    return hits / k

def recall_at_k(pred_scores, true_items, k=10):
    idx = np.argsort(-pred_scores)[:k]
    hits = sum(1 for i in idx if i in true_items)
    if len(true_items) == 0:
        return 0.0
    return hits / len(true_items)

def dcg_at_k(pred_scores, true_items, k=10):
    idx = np.argsort(-pred_scores)[:k]
    dcg = 0.0
    for i, item in enumerate(idx):
        rel = 1.0 if item in true_items else 0.0
        dcg += (2**rel - 1) / np.log2(i + 2)
    return dcg

def ndcg_at_k(pred_scores, true_items, k=10):
    dcg = dcg_at_k(pred_scores, true_items, k)
    # ideal dcg (all relevant items in top positions)
    ideal_rels = [1.0] * min(len(true_items), k)
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += (2**rel - 1) / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0
