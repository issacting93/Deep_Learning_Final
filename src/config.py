from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FMA_SMALL_DIR = DATA_DIR / "fma_small"
FMA_METADATA_DIR = DATA_DIR / "fma_metadata"
MODELS_DIR = PROJECT_ROOT / "models"

# Audio constants
CLAP_SR = 48000
CLAP_DURATION_S = 10
CLAP_DURATION_SAMPLES = CLAP_SR * CLAP_DURATION_S  # 480000

# Processing
CLAP_BATCH_SIZE = 32

# GNN (Role 3 — Issac)
GNN_HIDDEN_DIM = 256
GNN_OUT_DIM    = 256
GNN_EPOCHS     = 30

# Device selection
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "cpu"  # MPS has incomplete op support for CLAP
    return "cpu"

DEVICE = get_device()
