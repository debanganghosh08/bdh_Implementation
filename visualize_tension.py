# visualize_tension.py
# ============================
# Visualize BDH Tension Curve
# ============================

#import torch
#import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
#from sentence_transformers import SentenceTransformer
#from pathlib import Path
#
#from models.bdh_kernel import BDHProbe
#
#
## ----------------------------
## CONFIG
## ----------------------------
#
#DATA_DIR = Path("data")
#CHUNK_SIZE = 512
#OVERLAP = 128
#
#EXAMPLE_INDEX = 0   # change to visualize other samples
#
#
## ----------------------------
## LOAD MODELS
## ----------------------------
#
#embedder = SentenceTransformer("all-MiniLM-L6-v2")
#bdh = BDHProbe()
#
#
## ----------------------------
## HELPERS
## ----------------------------
#
#def load_novel(book_name):
#    return (DATA_DIR / f"{book_name}.txt").read_text(
#        encoding="utf-8", errors="ignore"
#    )
#
#def chunk_text(text):
#    tokens = text.split()
#    step = CHUNK_SIZE - OVERLAP
#    for i in range(0, len(tokens), step):
#        yield " ".join(tokens[i:i + CHUNK_SIZE])
#
#def embed(text):
#    return embedder.encode(text, convert_to_tensor=True).unsqueeze(0)
#
#
## ----------------------------
## MAIN
## ----------------------------
#
#def main():
#    df = pd.read_csv(DATA_DIR / "train.csv")
#    row = df.iloc[EXAMPLE_INDEX]
#
#    novel = load_novel(row["book_name"])
#    backstory = row["content"]
#
#    tensions = []
#
#    with torch.no_grad():
#        rho = bdh.imprint_backstory(embed(backstory))
#        x = torch.zeros(1, bdh.n)
#
#        for chunk in chunk_text(novel):
#            v = embed(chunk)
#            x, rho, tension = bdh.calculate_step(x, rho, v)
#            tensions.append(tension.item())
#
#    tensions = np.array(tensions)
#
#    plt.figure(figsize=(10, 4))
#    plt.plot(tensions)
#    plt.title("BDH Tension Curve Across Narrative")
#    plt.xlabel("Narrative Chunk Index")
#    plt.ylabel("Tension")
#    plt.grid(True)
#    plt.show()
#
#
#if __name__ == "__main__":
#    main()

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("train_tension.csv")

features = [
    "avg_tension",
    "max_tension",
    "std_tension",
    "slope",
    "early_late_ratio",
    "num_spikes"
]

df[features].hist(figsize=(12, 8), bins=30)
plt.suptitle("Distribution of BDH Tension Features")
plt.show()
