"""Select GSM8K cluster representatives and show them on a UMAP scatter plot."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from umap import UMAP

from experiments import gsm8k


def main():
    records = gsm8k()
    vectors = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode(
        [r["question"] for r in records], normalize_embeddings=True, show_progress_bar=True)
    clusters = KMeans(n_clusters=50, random_state=0, n_init=10).fit(vectors)
    selected = []
    for label, center in enumerate(clusters.cluster_centers_):
        members = np.flatnonzero(clusters.labels_ == label)
        selected.append(int(members[np.linalg.norm(vectors[members] - center, axis=1).argmin()]))
    Path("representatives.json").write_text(json.dumps([records[i] for i in selected], indent=2))
    xy = UMAP(random_state=0, metric="cosine", n_neighbors=15, min_dist=0.1).fit_transform(vectors)
    plt.scatter(*xy.T, c=clusters.labels_, s=3, cmap="tab20")
    plt.scatter(*xy[selected].T, c="gold", marker="D", s=25, edgecolors="black")
    plt.title("GSM8K: 50 representatives (gold). UMAP distorts distances.")
    plt.savefig("clusters.png")
    plt.show()


if __name__ == "__main__":
    main()
