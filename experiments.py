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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["prompt", "position", "length", "gsm8k", "injection"])
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("injection_cases.example.json"),
                        help="JSON case file for the injection experiment")
    parser.add_argument("--prompt", default="What is 12 times 15?")
    parser.add_argument("--tokens", type=int, default=125, help="Output cap for non-injection experiments")
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--pause-max", type=int, default=32, help="Inject after a uniform random 1..N tokens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true")
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
        cases = json.loads(args.cases.read_text())["cases"]
        for case in cases:
            for trial in range(args.trials):
                rows.append(dict(case=case, trial=trial, **exp.inject(
                    case["question"], case["injection"], args.pause_max)))
    print(json.dumps(dict(model=MODEL, settings={**vars(args), "cases": str(args.cases)}, results=rows), indent=2))
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
