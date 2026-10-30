"""Central configuration. Everything can be overridden with environment variables
(or a backend/.env file loaded by app.py). No secrets are needed for this project."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(os.getenv("PAPERMIND_DATA", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "indexes"
DB_PATH = DATA_DIR / "papermind.db"

# Fine-tuned retrieval model (produced by the Colab notebook). If the folder does not
# exist the app falls back to the pretrained base model, so it still runs.
MODEL_DIR = Path(os.getenv("PAPERMIND_MODEL_DIR", BASE_DIR / "models" / "papermind-minilm"))
BASE_MODEL = os.getenv("PAPERMIND_BASE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Extractive reader used to pull an answer span out of retrieved passages.
READER_MODEL = os.getenv("PAPERMIND_READER_MODEL", "deepset/minilm-uncased-squad2")

# Used only when the model folder has no papermind_config.json (the notebook writes one).
DEFAULT_RETRIEVAL_THRESHOLD = float(os.getenv("PAPERMIND_THRESHOLD", "0.30"))

MAX_UPLOAD_MB = int(os.getenv("PAPERMIND_MAX_UPLOAD_MB", "25"))
MAX_PAGES = int(os.getenv("PAPERMIND_MAX_PAGES", "80"))
MAX_QUESTION_CHARS = 500

CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "PAPERMIND_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]
