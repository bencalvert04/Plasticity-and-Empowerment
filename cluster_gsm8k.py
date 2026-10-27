"""Select GSM8K questions nearest the centers of embedding clusters."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


def representatives(embeddings, count, seed):
    if not 1 <= count <= len(embeddings):
        raise ValueError("Cluster count must be between 1 and the dataset size.")
    clustering = KMeans(n_clusters=count, random_state=seed, n_init=10).fit(embeddings)
    selected = []
    for cluster, center in enumerate(clustering.cluster_centers_):
        members = np.flatnonzero(clustering.labels_ == cluster)
        if not len(members):
            raise ValueError("An empty cluster was found; reduce the cluster count.")
        distances = np.linalg.norm(embeddings[members] - center, axis=1)
        nearest = int(members[distances.argmin()])
        selected.append((nearest, members.tolist(), float(distances.min())))
    return selected


def main(count=50, seed=0, output="gsm8k_clustered_50.json"):
    records = []
    for split in ("train", "test"):
        url = f"https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl"
        with urlopen(url, timeout=30) as response:
            records.extend(dict(json.loads(line), id=f"{split}:{i}")
                           for i, line in enumerate(response) if line.strip())
    if not 1 <= count <= len(records):
        raise ValueError("Cluster count must be between 1 and the dataset size.")
    torch.set_num_threads(4)
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(model_id, device=device)
    texts = [record["question"] for record in records]
    lengths = [len(ids) for ids in model.tokenizer(texts, truncation=False)["input_ids"]]
    truncated = sum(length > model.max_seq_length for length in lengths)
    print(f"Embedding {len(records)} questions on {device}; {truncated} exceed the embedding token limit.", flush=True)
    embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=True)
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain nonfinite values.")
    print(f"Clustering into {count} groups...", flush=True)
    questions = []
    for cluster, (index, members, distance) in enumerate(representatives(embeddings, count, seed)):
        questions.append(dict(records[index], cluster=cluster, cluster_size=len(members),
                              weight=len(members) / len(records), distance_to_center=distance,
                              member_ids=[records[i]["id"] for i in members]))
    result = {
        "embedding_model": model_id,
        "embedding_model_revision": getattr(model[0].auto_model.config, "_commit_hash", None),
        "embedding_max_tokens": model.max_seq_length,
        "truncated_question_count": truncated,
        "dataset_size": len(records),
        "dataset_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
        "seed": seed,
        "method": "KMeans on normalized question embeddings; nearest member to each centroid",
        "note": "Cluster weights support approximate dataset summaries; centroid selection is not an unbiased random sample.",
        "questions": questions,
    }
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {count} representatives to {output}; cluster sizes range from "
          f"{min(q['cluster_size'] for q in questions)} to {max(q['cluster_size'] for q in questions)}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.clusters < 1:
        parser.error("--clusters must be positive.")
    main(args.clusters, args.seed, args.output or f"gsm8k_clustered_{args.clusters}.json")
