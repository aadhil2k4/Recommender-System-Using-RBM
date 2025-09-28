import torch
import joblib
from pathlib import Path

def save_model(model: torch.nn.Module, path: Path):
    torch.save(model.state_dict(), str(path))

def load_model_state(model: torch.nn.Module, path: Path, device="cpu"):
    state = torch.load(str(path), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def save_mapping(mapping: dict, path: Path):
    joblib.dump(mapping, path)

def load_mapping(path: Path):
    return joblib.load(path)
