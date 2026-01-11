# run_bdh.py
# ============================
# Multi-Feature BDH Inference
# ============================

import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path

from models.bdh_kernel import BDHProbe
from models.classifier import TensionClassifier


DATA_DIR = Path("data")
CHUNK_SIZE = 512
OVERLAP = 128

embedder = SentenceTransformer("all-MiniLM-L6-v2")
bdh = BDHProbe()
classifier = TensionClassifier()

classifier.load_state_dict(
    torch.load("tension_head.pt", map_location="cpu")
)
classifier.eval()


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


def extract_features(backstory, novel):
    tensions = []

    with torch.no_grad():
        rho = bdh.imprint_backstory(embed(backstory))
        x = torch.zeros(1, bdh.n)

        for chunk in chunk_text(novel):
            v = embed(chunk)
            x, rho, tension = bdh.calculate_step(x, rho, v)
            tensions.append(tension.item())

    tensions = np.array(tensions)
    half = len(tensions) // 2

    return torch.tensor([
        tensions.mean(),
        tensions.max(),
        tensions.std(),
        tensions[-1] - tensions[0],
        tensions[:half].mean() / (tensions[half:].mean() + 1e-6),
        np.sum(tensions > tensions.mean() + 2 * tensions.std())
    ], dtype=torch.float32).unsqueeze(0)


def main():
    df = pd.read_csv(DATA_DIR / "test.csv")
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        novel = load_novel(row["book_name"])
        feats = extract_features(row["content"], novel)
        prob = classifier(feats).item()
        pred = int(prob > 0.5)

        rows.append({
            "id": row["id"],
            "prediction": pred,
            "label": "Consistent" if pred else "Contradict"
        })

    pd.DataFrame(rows).to_csv("results.csv", index=False)
    print("✅ results.csv generated")


if __name__ == "__main__":
    main()
