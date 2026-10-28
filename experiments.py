"""Small entry point for the five experiments. Redirect stdout to save JSON."""
import argparse
import json
import random
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import torch

from core import ANSWER, MODEL, Experiment


def gsm8k():
    records = []
    for split in ("train", "test"):
        url = f"https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl"
        with urlopen(url, timeout=30) as response:
            records.extend(dict(json.loads(line), id=f"{split}:{i}") for i, line in enumerate(response))
    return records


def compact_result(row, experiment, tokenizer):
    """Save measurements and identifiers; keep token-level records behind --details."""
    compact = {key: row[key] for key in ("id", "trial", "length") if key in row}
    if "case" in row:
        compact.update(id=row["case"]["id"], correction=row["case"]["injection"], pause=row["pause"])
        if "skipped" in row:
            return dict(compact, skipped=row["skipped"])
        compact.update(scored_tokens=row["scored_tokens"], window_means=row["window_means"])
    elif "id" not in row:
        compact["prompt"] = row["prompt"]
    result = row.get("continuation", row)
    answer_tokens = [result["tokens"][i] for i in result["answer_indices"]]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip() if answer_tokens else None
    compact.update(answer=answer, complete=result["complete"],
                   token_count=len(result["tokens"]), answer_means=result["answer_means"])
    if experiment in ("position", "length"):
        compact["score_means"] = {name: np.mean(values, axis=0).tolist()
                                  for name, values in result["scores"].items()}
        compact["score_totals"] = {name: np.sum(values, axis=0).tolist()
                                   for name, values in result["scores"].items()}
    return compact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["prompt", "position", "length", "gsm8k", "injection"])
    parser.add_argument("--prompt", default="What is 12 times 15?")
    parser.add_argument("--tokens", type=int, default=125, help="Output cap for non-injection experiments")
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--pause-max", type=int, default=32, help="Inject after a uniform random 1..N tokens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--details", action="store_true", help="Save full text, token IDs and per-token scores")
    args = parser.parse_args()
    if min(args.tokens, args.sample, args.trials, args.pause_max) < 1:
        parser.error("Counts must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    exp = Experiment()
    prompts = json.loads(Path(__file__).with_name("prompts.json").read_text())
    rows = []
    if args.experiment == "prompt":
        rows = [dict(prompt=args.prompt, **exp.prompt(args.prompt + ANSWER, args.tokens))]
    elif args.experiment == "position":
        rows = [dict(prompt=p, **exp.prompt(p, args.tokens, greedy=True)) for p in prompts["short"]]
    elif args.experiment == "length":
        # Slice original token IDs directly, without decoding and retokenizing prefixes.
        before, after = exp.tokenizer.apply_chat_template(
            [{"role": "user", "content": "{PROMPT}"}], tokenize=False,
            add_generation_prompt=True).split("{PROMPT}")
        before, after = [exp.tokenizer.encode(s, add_special_tokens=False) for s in (before, after)]
        for length in range(5, 50, 2):
            for prompt in prompts["long"]:
                ids = exp.tokenizer.encode(prompt, add_special_tokens=False)[:length]
                result = exp.run({"full": before + ids + after, "input": before + after}, args.tokens, greedy=True)
                rows.append(dict(length=len(ids), prompt=prompt, **result))
    elif args.experiment == "gsm8k":
        for question in random.sample(gsm8k(), args.sample):
            for trial in range(args.trials):
                rows.append(dict(id=question["id"], prompt=question["question"], trial=trial,
                                 **exp.prompt(question["question"] + ANSWER, args.tokens)))
    else:
        cases = json.loads(Path(__file__).with_name("injection_cases.example.json").read_text())["cases"]
        for case in cases:
            for trial in range(args.trials):
                rows.append(dict(case=case, trial=trial, **exp.inject(
                    case["question"], case["injection"], args.pause_max)))
    saved = rows if args.details else [compact_result(row, args.experiment, exp.tokenizer) for row in rows]
    print(json.dumps(dict(model=MODEL, settings=vars(args),
                         metric_order=["entropy_difference_bits", "js_bits"], results=saved), indent=2))
    if args.plot:
        if args.experiment == "length":
            lengths = sorted({r["length"] for r in rows})
            plt.plot(lengths, [np.mean([sum(v[0] for v in r["scores"]["input"])
                                      for r in rows if r["length"] == n]) for n in lengths])
            plt.xlabel("Prompt tokens")
            plt.ylabel("Mean summed entropy reduction (bits)")
        else:
            results = [r.get("continuation", r) for r in rows if "skipped" not in r]
            if not results:
                return
            for name in results[0]["scores"]:
                n = min(len(r["tokens"]) for r in results)
                plt.plot(range(1, n + 1), np.mean([r["scores"][name][:n] for r in results], axis=0)[:, 0], label=name)
            plt.xlabel("Output token position (only positions reached by every response)")
            plt.ylabel("Mean entropy reduction (bits)")
            plt.legend()
        plt.axhline(0, color="gray")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
