"""Download the embedding model into ./models/ (run where internet/SSL works).

After download, copy the models/ folder to your work laptop and set in config.py:
    HF_HUB_OFFLINE = True
"""

from pathlib import Path

from sentence_transformers import SentenceTransformer

from config import BASE_DIR, EMBED_MODEL
from embed_utils import _apply_ssl_and_hub_settings

OUT_DIR = BASE_DIR / "models" / "all-MiniLM-L6-v2"


def main():
    _apply_ssl_and_hub_settings()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {EMBED_MODEL} -> {OUT_DIR}")
    model = SentenceTransformer(EMBED_MODEL)
    model.save(str(OUT_DIR))
    print("Done. Set HF_HUB_OFFLINE = True in config.py on this machine.")


if __name__ == "__main__":
    main()
