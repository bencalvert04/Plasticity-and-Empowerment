"""Entropy-difference proxies on random GSM8K questions."""

import argparse
import json
import random
import os
import certifi
from urllib.request import urlopen
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from input_and_output_directed_information import generate_and_score
from answer_token_scores import score_numeric_answer


def main(max_new_tokens=None, sample=10, trials=1, answer_only_scores=False, js_scores=False, output=None, seed=None):
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    if max_new_tokens is not None and max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    output_dir = Path(output) if output else None
    if output_dir and output_dir.exists():
        raise FileExistsError("Choose a new output directory; existing records will not be overwritten")
    if not 1 <= sample <= 8000:
        raise ValueError("sample must be between 1 and 8000.")
    if trials < 1:
        raise ValueError("trials must be a positive integer.")
    if not torch.backends.mps.is_available():
        raise RuntimeError("This experiment requires an Apple GPU with MPS support.")
    source_records = []
    for split in ("train", "test"):
        url = f"https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl"
        with urlopen(url, timeout=30) as response:
            source_records.extend({"id": f"{split}:{i}", "question": json.loads(line)["question"]}
                                  for i, line in enumerate(response) if line.strip())
    records = random.sample(source_records, sample)
    questions = [r["question"] for r in records]
    torch.set_num_threads(4)
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16,
        attn_implementation="sdpa",
    ).to("mps").eval()

    if output_dir:
        output_dir.mkdir(parents=True)
        (output_dir / "metadata.json").write_text(json.dumps({
            "model": model_id, "model_commit": getattr(model.config, "_commit_hash", None),
            "torch": torch.__version__, "seed": seed, "trials": trials,
            "max_new_tokens": max_new_tokens, "answer_only_scores": answer_only_scores,
            "js_scores": js_scores, "questions": records,
            "sampling": "full softmax, temperature 1; GPU reproducibility not guaranteed",
            "interpretation": "Entropy and JS context-sensitivity proxies, not formal DI or empowerment"
        }, indent=2))
    all_dir_inf_ItoO, all_dir_inf_POtoFO = [], []
    for i, question in enumerate(questions, start=1):
        print(f"\nQuestion {i}: {question}", flush=True)
        for trial in range(1, trials + 1):
            print(f"\nTrial {trial}/{trials}", flush=True)
            scored = generate_and_score(
                model, tokenizer, question, max_new_tokens,
                return_details=answer_only_scores or js_scores or bool(output_dir),
                compute_js=js_scores,
            )
            response, dir_inf_ItoO, dir_inf_POtoFO = scored[:3]
            all_dir_inf_ItoO.append(sum(dir_inf_ItoO))
            all_dir_inf_POtoFO.append(sum(dir_inf_POtoFO))
            output_token_count = len(dir_inf_ItoO)  # One scored term per token, including EOS.
            print(f"Response: {response}")
            print(f"Output-token count (including EOS): {output_token_count}")
            print(f"directed information (input to output, given previous output): {sum(dir_inf_ItoO):.6f} bits")
            print(f"directed information (previous output to next output, given input): {sum(dir_inf_POtoFO):.6f} bits")
            print(f"dir_inf_ItoO per token: {sum(dir_inf_ItoO) / output_token_count:.6f} bits/token")
            print(f"dir_inf_POtoFO per token: {sum(dir_inf_POtoFO) / output_token_count:.6f} bits/token")
            details = scored[3] if len(scored) == 4 else None
            answer = None
            if js_scores:
                print(f"Input-context JS: {sum(details['input_js_bits']) / output_token_count:.6f} bits/token")
                print(f"Output-history JS: {sum(details['history_js_bits']) / output_token_count:.6f} bits/token")
            if answer_only_scores:
                answer = score_numeric_answer(
                    tokenizer, details["token_ids"], dir_inf_ItoO, dir_inf_POtoFO,
                    details["ended_with_eos"],
                )
                print(f"Answer-token scoring: {answer['status']}")
                if answer["status"] == "ok":
                    print(f"Final numerical answer: {answer['answer']}")
                    print(f"Answer token count: {answer['token_count']}; output positions (1-based): {[i + 1 for i in answer['token_indices']]}")
                    print(f"Input -> answer entropy proxy: {answer['input_bits_per_token']:.6f} bits/token")
                    print(f"Previous output -> answer entropy proxy (not established empowerment): {answer['history_bits_per_token']:.6f} bits/token")
                    if js_scores:
                        indices = answer["token_indices"]
                        for name in ("input", "history"):
                            value = sum(details[f"{name}_js_bits"][j] for j in indices) / len(indices)
                            answer[f"{name}_js_bits_per_token"] = value
                            print(f"Answer {name}-context JS: {value:.6f} bits/token")
            if output_dir:
                saved = {"question_id": records[i-1].get("id", str(i)), "trial": trial,
                         "question": question, "response": response, "output_token_count": output_token_count,
                         "input_entropy_bits": dir_inf_ItoO, "history_entropy_bits": dir_inf_POtoFO,
                         "input_entropy_mean": sum(dir_inf_ItoO) / output_token_count,
                         "history_entropy_mean": sum(dir_inf_POtoFO) / output_token_count,
                         "answer_scores": answer, **details}
                with (output_dir / "trials.jsonl").open("a") as file:
                    file.write(json.dumps(saved) + "\n")

    print(f"\nMaximum dir_inf_ItoO across all question trials: {max(all_dir_inf_ItoO):.6f} bits")
    print(f"Maximum dir_inf_POtoFO across all question trials: {max(all_dir_inf_POtoFO):.6f} bits")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=10, help="Number of questions, 1–8000 (default: 10).")
    parser.add_argument("--trials", type=int, default=1, help="Responses per question (positive integer; default: 1).")
    parser.add_argument("--answer-only-scores", action="store_true", help="Also average both entropy proxies over final numerical-answer tokens; preserves full-response scores.")
    parser.add_argument("--js-scores", action="store_true", help="Also compare full next-token distributions with Jensen-Shannon divergence.")
    parser.add_argument("--output", help="New directory for metadata and per-trial JSONL records.")
    parser.add_argument("--seed", type=int, help="Seed question and response sampling.")
    parser.add_argument("--max-new-tokens", type=int, help="Optional output cap; default remains uncapped.")
    args = parser.parse_args()
    if not 1 <= args.sample <= 8000:
        parser.error("--sample must be between 1 and 8000.")
    if args.trials < 1:
        parser.error("--trials must be a positive integer.")
    main(sample=args.sample, trials=args.trials, answer_only_scores=args.answer_only_scores, js_scores=args.js_scores, output=args.output, seed=args.seed, max_new_tokens=args.max_new_tokens)
