"""Select one representative question per GSM8K embedding cluster."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


def main(count=50, seed=0, output=None):
    records = []
    for split in ("train", "test"):
        url = f"https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl"
        with urlopen(url, timeout=30) as response:
            records.extend(dict(json.loads(line), id=f"{split}:{i}")
                           for i, line in enumerate(response) if line.strip())
    if not 1 <= count <= len(records):
        raise ValueError("Cluster count must be between 1 and the dataset size.")

    torch.set_num_threads(4)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_id, device=device)
    embeddings = model.encode([r["question"] for r in records], batch_size=64,
                              normalize_embeddings=True, show_progress_bar=True)
    clusters = KMeans(n_clusters=count, random_state=seed, n_init=10).fit(embeddings)
    questions = []
    for cluster, center in enumerate(clusters.cluster_centers_):
        members = np.flatnonzero(clusters.labels_ == cluster)
        if not len(members):
            raise ValueError("Empty cluster; reduce the cluster count.")
        nearest = members[np.linalg.norm(embeddings[members] - center, axis=1).argmin()]
        questions.append(dict(records[nearest], cluster=cluster, cluster_size=len(members),
                              weight=len(members) / len(records)))
    output = output or f"gsm8k_clustered_{count}.json"
    result = dict(embedding_model=model_id, seed=seed, dataset_size=len(records), questions=questions)
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {count} questions to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()
    main(args.clusters, args.seed, args.output)
