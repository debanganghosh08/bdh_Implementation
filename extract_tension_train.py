# extract_tension_train.py
# ============================
# Extract Multi-Feature BDH Tension (TRAIN SET)
# ============================

import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path

from models.bdh_kernel import BDHProbe


DATA_DIR = Path("data")
CHUNK_SIZE = 512
OVERLAP = 128

embedder = SentenceTransformer("all-MiniLM-L6-v2")
bdh = BDHProbe()


def load_novel(book_name):
    return (DATA_DIR / f"{book_name}.txt").read_text(
        encoding="utf-8", errors="ignore"
    )


def chunk_text(text):
    tokens = text.split()
    step = CHUNK_SIZE - OVERLAP
    for i in range(0, len(tokens), step):
        yield " ".join(tokens[i:i + CHUNK_SIZE])


def embed(text):
    return embedder.encode(text, convert_to_tensor=True).unsqueeze(0)


def extract_tension_features(backstory, novel):
    tensions = []

    with torch.no_grad():
        rho = bdh.imprint_backstory(embed(backstory))
        x = torch.zeros(1, bdh.n)

        for chunk in chunk_text(novel):
            v = embed(chunk)
            x, rho, tension = bdh.calculate_step(x, rho, v)
            tensions.append(tension.item())

    tensions = np.array(tensions)

    if len(tensions) == 0:
        return None

    half = len(tensions) // 2

    return {
        "avg_tension": tensions.mean(),
        "max_tension": tensions.max(),
        "std_tension": tensions.std(),
        "slope": tensions[-1] - tensions[0],
        "early_late_ratio": tensions[:half].mean() /
                            (tensions[half:].mean() + 1e-6),
        "num_spikes": int(
            np.sum(tensions > tensions.mean() + 2 * tensions.std())
        )
    }


def main():
    df = pd.read_csv(DATA_DIR / "train.csv")
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        novel = load_novel(row["book_name"])
        feats = extract_tension_features(row["content"], novel)

        if feats is None:
            continue

        feats["label"] = row["label"]
        rows.append(feats)

    pd.DataFrame(rows).to_csv("train_tension.csv", index=False)
    print("✅ train_tension.csv generated (multi-feature)")


if __name__ == "__main__":
    main()
