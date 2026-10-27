"""Map GSM8K embeddings with UMAP and highlight an existing clustered selection."""

import argparse
import hashlib
import html
import json
import textwrap
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import plotly.graph_objects as go
import torch
from sentence_transformers import SentenceTransformer
from umap import UMAP


def main(selection='gsm8k_clustered_50.json', output='gsm8k_umap.html'):
    saved = json.loads(Path(selection).read_text())
    records = []
    for split in ('train', 'test'):
        url = f'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl'
        with urlopen(url, timeout=30) as response:
            records.extend(dict(json.loads(line), id=f'{split}:{i}')
                           for i, line in enumerate(response) if line.strip())
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    if saved.get('dataset_sha256', digest) != digest:
        raise ValueError('Dataset differs from the saved selection.')
    indices = {r['id']: i for i, r in enumerate(records)}
    selected = saved['questions']
    selected_indices = [indices[q['id']] for q in selected]
    assert all(records[i]['question'] == q['question'] for i, q in zip(selected_indices, selected))
    torch.set_num_threads(4)
    model = SentenceTransformer(saved['embedding_model'],
                               revision=saved.get('embedding_model_revision'),
                               device='mps' if torch.backends.mps.is_available() else 'cpu')
    embeddings = model.encode([r['question'] for r in records], batch_size=64,
                              normalize_embeddings=True, show_progress_bar=True)
    # Use original memberships when present. Compact selections need KMeans rerun.
    if all('member_ids' in q for q in selected):
        labels = np.full(len(records), -1)
        for q in selected:
            labels[[indices[i] for i in q['member_ids']]] = q['cluster']
        assert (labels >= 0).all()
    else:
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=len(selected), random_state=saved['seed'], n_init=10).fit_predict(embeddings)
        if any(labels[i] != q['cluster'] for i, q in zip(selected_indices, selected)):
            raise ValueError('Recomputed clusters differ from the saved selection.')
    print('Projecting embeddings into 2D with UMAP...', flush=True)
    points = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine',
                  random_state=0, n_jobs=1).fit_transform(embeddings)
    hover = [f"{r['id']} | Cluster {labels[i] + 1}<br>" + '<br>'.join(
        html.escape(line) for line in textwrap.wrap(r['question'], 85)) for i, r in enumerate(records)]
    fig = go.Figure()
    for cluster in sorted(set(labels)):
        members = np.flatnonzero(labels == cluster)
        fig.add_trace(go.Scattergl(x=points[members, 0], y=points[members, 1],
            mode='markers', name=f'Cluster {cluster + 1} ({len(members)})',
            text=[hover[i] for i in members], hovertemplate='%{text}<extra></extra>',
            marker=dict(size=5, opacity=0.55, color=f'hsl({cluster * 137.508 % 360},65%,48%)')))
    fig.add_trace(go.Scattergl(x=points[selected_indices, 0], y=points[selected_indices, 1],
        mode='markers', name=f'{len(selected)} selected questions',
        text=[hover[i] for i in selected_indices], hovertemplate='<b>SELECTED</b><br>%{text}<extra></extra>',
        marker=dict(size=13, symbol='diamond', color='#ffe066', line=dict(width=2, color='#172033'))))
    fig.update_layout(template='plotly_white', height=850,
        title=dict(text=f'GSM8K: {len(records):,} questions, {len(selected)} representatives<br>'
                   '<sup>Gold diamonds = selected questions. Hover to read; scroll to zoom; click legend to filter.</sup>'),
        xaxis_title='UMAP 1', yaxis_title='UMAP 2', legend=dict(title='Clusters', itemsizing='constant'),
        margin=dict(t=130, b=90), dragmode='pan',
        updatemenus=[dict(type='buttons', direction='left', x=0, y=1.08, buttons=[
            dict(label='All questions', method='update', args=[{'visible': [True] * len(fig.data)}]),
            dict(label='Selected only', method='update', args=[{'visible': [False] * (len(fig.data)-1) + [True]}])])],
        annotations=[dict(x=0, y=-0.10, xref='paper', yref='paper', showarrow=False,
            text='Clusters were formed in the original embedding space. UMAP distorts distances; islands are not proof of distinct problem types.')])
    fig.write_html(output, include_plotlyjs=True, config={'scrollZoom': True, 'displaylogo': False})
    Path(output).with_suffix('.json').write_text(json.dumps(dict(
        selection=str(selection), dataset_sha256=digest, umap=dict(seed=0, n_neighbors=15, min_dist=0.1, metric='cosine'),
        points=[dict(id=r['id'], cluster=int(labels[i]), x=float(points[i, 0]), y=float(points[i, 1]),
                     selected=i in selected_indices) for i, r in enumerate(records)]), indent=2) + '\n')
    print(f'Saved interactive map to {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', default='gsm8k_clustered_50.json')
    parser.add_argument('--output', default='gsm8k_umap.html')
    args = parser.parse_args()
    main(args.selection, args.output)
