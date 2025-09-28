import torch
from torch.utils.data import DataLoader

from src.config import DEFAULTS, RBM_CKPT, USERITEM_BINARY
from src.models.rbm import RBM
from src.dataset import UserItemDataset


def train_rbm(
    ratings_npz_path=USERITEM_BINARY,
    n_hidden=256,
    k=1,
    lr=1e-3,
    batch_size=128,
    epochs=20,
    weight_decay=1e-4,
    device="cpu"
):
    ds = UserItemDataset(ratings_npz_path)
    n_items = ds.n_items
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = RBM(n_visible=n_items, n_hidden=n_hidden, k=k).to(device)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            v_batch, mask = batch  # mask unused for binarized RBM (expect 0/1)
            v_batch = (v_batch > 0).float().to(device)  # binarize on the fly
            loss = model.contrastive_divergence_update(
                v_batch, lr=lr, weight_decay=weight_decay
            )
            epoch_loss += loss.item() * v_batch.size(0)
        print(f"[RBM] Epoch {epoch}/{epochs}  avg_loss={epoch_loss / len(ds):.6f}")

    torch.save(model.state_dict(), RBM_CKPT)
    print(f"Saved RBM to {RBM_CKPT}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", default=USERITEM_BINARY, help="Path to user-item binary npz")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["rbm"]["epochs"])
    args = parser.parse_args()

    train_rbm(
        ratings_npz_path=args.ratings,
        n_hidden=DEFAULTS["rbm"]["n_hidden"],
        k=DEFAULTS["rbm"]["cd_k"],
        lr=DEFAULTS["rbm"]["lr"],
        batch_size=DEFAULTS["rbm"]["batch_size"],
        epochs=args.epochs,
    )
