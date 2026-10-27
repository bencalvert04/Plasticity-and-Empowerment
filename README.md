# Plasticity and Empowerment

Experiments on how an initial prompt changes a language model's next-token entropy.

- `directed_information_convergence_to_zero.py`: runs Pythia-70m on 20 fixed prompts and plots the mean per-token entropy change. The current output limit is 125 tokens, with early stopping at EOS.
- `effect_of_prompt_length_on_directed_information.py`: uses five fixed long prompts and prints and plots averages for input lengths from 5 through 49 tokens in steps of 2. Each average is the mean summed entropy change across the five prompt prefixes of that length. Every run uses the same 125-token output limit, with early stopping at EOS. Input lengths exclude BOS.

```sh
python3 -m pip install -r requirements.txt
python3 directed_information_convergence_to_zero.py
python3 effect_of_prompt_length_on_directed_information.py
```

The first run downloads the model. Generation currently uses greedy decoding on CPU with float64 parameters. A commented sampling alternative is included. The graph averages only positions reached by every response.

At each output position, the score is:

```text
H(next token | BOS, previous output)
  - H(next token | BOS, prompt, previous output)
```

Entropies are in bits. Despite the experiment names, this is prompt-induced entropy reduction, not a formal directed-information estimator or a proof of convergence to zero.

### Embedding-based question selection

`cluster_gsm8k.py` embeds all 8,792 training and test questions using
`sentence-transformers/all-MiniLM-L6-v2`, then applies KMeans to normalized
embeddings with seed 0 and 10 initializations. It selects the nearest member of
each cluster to its center and saves the questions, reference answers, IDs,
cluster sizes and weights to `gsm8k_clustered_50.json` by default.
Only question text is embedded, using the embedding model’s default token limit.

Install the additional dependencies and call clustering from the experiment:

```sh
python3 -m pip install -r requirements-clustering.txt
python3 gsm8k_directed_information.py --clusters 50 --trials 5
```

To create the subset separately, or reuse the exact saved selection:

```sh
python3 cluster_gsm8k.py --clusters 50
python3 gsm8k_directed_information.py --questions-file gsm8k_clustered_50.json --trials 5
```

`--clusters`, `--questions-file`, and `--sample` are mutually exclusive.
Without these options, the experiment still samples 10 questions randomly.
The clustering seed does not seed generated responses. Cluster weights are
saved for future approximate dataset summaries; the experiment still reports
individual trial scores and maxima, without averaging or weighting them.
Centroid representatives are not an unbiased random sample, and semantic
coverage does not guarantee coverage of mathematical difficulty.

### Interactive UMAP map

Open `gsm8k_umap.html` in a browser to explore all 8,792 questions. The 50
saved representatives are gold diamonds. Hover to read questions, scroll to
zoom, drag to pan, and use the legend or Selected only button to filter.
The HTML includes Plotly and works offline after downloading.

To rebuild the map from the saved selection:

```sh
python3 -m pip install -r requirements-visualization.txt
python3 visualize_gsm8k.py --selection gsm8k_clustered_50.json
```

The map uses question embeddings and original saved cluster memberships,
with UMAP seed 0, cosine distance, 15 neighbors, and minimum distance 0.1.
For compact selection files without memberships, KMeans is rerun with the
saved seed. The selected IDs remain those in the selection file.
Projection coordinates and selected flags are also saved in `gsm8k_umap.json`.
UMAP is a distorted 2D view, not an exact topology or a measure of difficulty.
