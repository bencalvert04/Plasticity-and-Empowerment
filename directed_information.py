"""Monte Carlo conditional DI for an explicit finite pool of source token blocks."""
import argparse
import json
import math
from pathlib import Path

import torch

from core import ANSWER, Experiment, MODEL, entropy


def conditional_information(probabilities, log_weights):
    """I(source; next token | observed prefix), marginalizing over source values."""
    weights = log_weights.softmax(0)
    mixture = (weights[:, None] * probabilities).sum(0)
    ratio = probabilities / mixture.clamp_min(torch.finfo(probabilities.dtype).tiny)
    divergences = torch.special.xlogy(probabilities, ratio).sum(1) / math.log(2)
    return float((weights * divergences).sum().clamp_min(0))


def estimate(exp, inputs, reasonings, marker, trials=32, limit=16, seed=42, prior=None):
    """Joint law: supplied p(X,R), then model-generated A given X,R and the marker.

    Rollouts sample both sources from that joint law. Each answer prefix updates
    p(X | R, prefix) and p(R | X, prefix) by Bayes' rule. All comparisons replay
    the same sampled answer tokens. EOS is included, then treated as absorbing.
    """
    if min(trials, limit, len(inputs), len(reasonings)) < 1:
        raise ValueError("Source pools, trial count and answer limit must be positive")
    for blocks in (inputs, reasonings):
        if len({tuple(ids) for ids in blocks}) != len(blocks):
            raise ValueError("Source blocks must be distinct; combine duplicate prior weights")
    joint = torch.ones(len(inputs), len(reasonings)) if prior is None else torch.tensor(prior).float()
    if (joint.shape != (len(inputs), len(reasonings)) or not torch.isfinite(joint).all()
            or (joint < 0).any() or joint.sum() <= 0):
        raise ValueError("Prior must be a finite, nonnegative question-by-reasoning weight matrix")
    joint /= joint.sum()
    torch.manual_seed(seed)
    draws = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(trials):
        flat = torch.multinomial(joint.flatten(), 1, generator=draws).item()
        x, r = divmod(flat, len(reasonings))
        xs = torch.where(joint[:, r] > 0)[0].tolist()
        rs = torch.where(joint[x] > 0)[0].tolist()
        pairs = sorted({(i, r) for i in xs} | {(x, j) for j in rs})
        states = {(i, j): exp.predict(inputs[i] + reasonings[j] + marker) for i, j in pairs}
        device = states[x, r][0].device
        wx = joint[xs, r].to(device).log()
        wr = joint[x, rs].to(device).log()
        tokens, scores = [], []
        for step in range(limit):
            px = torch.stack([states[i, r][0] for i in xs])
            pr = torch.stack([states[x, j][0] for j in rs])
            scores.append([conditional_information(px, wx), conditional_information(pr, wr)])
            token = torch.multinomial(states[x, r][0], 1).item()
            tokens.append(token)
            if token in exp.eos:
                break
            wx += px[:, token].log()
            wr += pr[:, token].log()
            if step + 1 < limit:
                states = {pair: exp.predict([token], cache) for pair, (_, cache) in states.items()}
        total = torch.tensor(scores).sum(0)
        rows.append(dict(input_index=x, reasoning_index=r, answer_tokens=tokens,
                         ended_with_eos=tokens[-1] in exp.eos, per_token_bits=scores,
                         total_bits=total.tolist(), bits_per_token=(total / len(tokens)).tolist()))
    totals = torch.tensor([row['total_bits'] for row in rows])
    means = torch.tensor([row['bits_per_token'] for row in rows])
    bounds = torch.stack([entropy(joint) - entropy(joint.sum(0)),
                          entropy(joint) - entropy(joint.sum(1))])
    return dict(metric_order=['input_to_answer_given_reasoning', 'reasoning_to_answer_given_input'],
                joint_prior=joint.tolist(), mean_di_bits=totals.mean(0).tolist(),
                source_entropy_bounds_bits=bounds.tolist(),
                exceeds_source_bound=(totals.mean(0) > bounds + 1e-5).tolist(),
                monte_carlo_se_bits=(totals.std(0) / math.sqrt(trials)).tolist() if trials > 1 else None,
                mean_bits_per_token=means.mean(0).tolist(),
                capped_trials=sum(not row['ended_with_eos'] for row in rows), trials=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', required=True, help='JSON with questions and reasoning_token_ids')
    parser.add_argument('--trials', type=int, default=32)
    parser.add_argument('--answer-tokens', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--details', action='store_true', help='Include token blocks and per-trial records')
    args = parser.parse_args()
    sources = json.loads(Path(args.sources).read_text())
    if sources['model'] != MODEL:
        parser.error('Reasoning token IDs must come from the configured model tokenizer')
    exp = Experiment()
    inputs = [exp.chat(q + ANSWER) for q in sources['questions']]
    marker = exp.tokenizer.encode('\nFinal answer:', add_special_tokens=False)
    result = estimate(exp, inputs, sources['reasoning_token_ids'], marker, args.trials,
                      args.answer_tokens, args.seed, sources.get('joint_prior'))
    result.update(model=MODEL, settings=vars(args), sources=sources, answer_marker_ids=marker,
                  target='Answer-stage tokens including EOS, capped at answer-tokens; no numeric filtering')
    if args.details:
        result['input_token_ids'] = inputs
        for row in result['trials']:
            x, r = row['input_index'], row['reasoning_index']
            b = len(inputs[x])
            d = b + len(sources['reasoning_token_ids'][r])
            e = d + len(marker)
            row['inclusive_token_ranges'] = dict(input=[0, b-1], reasoning=[b, d-1],
                                                  answer_marker=[d, e-1], answer=[e, e+len(row['answer_tokens'])-1])
            row['answer_text'] = exp.tokenizer.decode(row['answer_tokens'], skip_special_tokens=True)
    else:
        result.pop('trials')
        result.pop('sources')
        result.pop('answer_marker_ids')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
