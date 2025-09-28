import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from ..data.dataset import UserItemDataset
from ..models.autoencoder import Autoencoder, masked_mse_loss
from ..config import DEFAULTS, AE_CKPT, RATINGS_NPZ

def train_ae(ratings_npz_path, hidden_dims=None, latent_dim=128, lr=1e-3, batch_size=128, epochs=20, dropout=0.3, device="cpu"):
    ds = UserItemDataset(ratings_npz_path)
    n_items = ds.n_items
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    hidden_dims = hidden_dims or DEFAULTS["ae"]["hidden_dims"]
    model = Autoencoder(n_items=n_items, hidden_dims=hidden_dims, latent_dim=latent_dim, dropout=dropout).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        for batch in loader:
            x, mask = batch
            x = x.to(device)
            mask = mask.to(device)
            pred = model(x)
            loss = masked_mse_loss(pred, x, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * x.size(0)
        print(f"[AE] Epoch {epoch}/{epochs}  avg_masked_mse={epoch_loss / len(ds):.6f}")
    torch.save(model.state_dict(), AE_CKPT)
    print(f"Saved AE to {AE_CKPT}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", default="../../data/processed/userItem_ratings.npz")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["ae"]["epochs"])
    args = parser.parse_args()
    train_ae(ratings_npz_path=args.ratings, hidden_dims=DEFAULTS["ae"]["hidden_dims"],
             latent_dim=DEFAULTS["ae"]["latent_dim"], lr=DEFAULTS["ae"]["lr"],
             batch_size=DEFAULTS["ae"]["batch_size"], epochs=args.epochs)
