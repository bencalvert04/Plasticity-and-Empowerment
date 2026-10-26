"""entropy-difference proxies on a random GSM8K sample."""

import argparse
import json
import random
from urllib.request import urlopen

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from input_and_output_directed_information import generate_and_score


def main(max_new_tokens=None, sample=10, trials=1):
    if not 1 <= sample <= 8000:
        raise ValueError("sample must be between 1 and 8000.")
    if trials < 1:
        raise ValueError("trials must be a positive integer.")
    if not torch.backends.mps.is_available():
        raise RuntimeError("This experiment requires an Apple GPU with MPS support.")
    questions = []
    for split in ("train", "test"):
        url = f"https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl"
        with urlopen(url, timeout=30) as response:
            questions.extend(json.loads(line)["question"] for line in response if line.strip())
    questions = random.sample(questions, sample)

    torch.set_num_threads(4)
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16,
        attn_implementation="sdpa",
    ).to("mps").eval()

    all_dir_inf_ItoO, all_dir_inf_POtoFO = [], []
    for i, question in enumerate(questions, start=1):
        print(f"\nQuestion {i}: {question}", flush=True)
        for trial in range(1, trials + 1):
            print(f"\nTrial {trial}/{trials}", flush=True)
            response, dir_inf_ItoO, dir_inf_POtoFO = generate_and_score(
                model, tokenizer, question, max_new_tokens,
            )
            all_dir_inf_ItoO.append(sum(dir_inf_ItoO))
            all_dir_inf_POtoFO.append(sum(dir_inf_POtoFO))
            output_token_count = len(dir_inf_ItoO)  # One scored term per token, including EOS.
            print(f"Response: {response}")
            print(f"Output-token count (including EOS): {output_token_count}")
            print(f"directed information (input to output, given previous output): {sum(dir_inf_ItoO):.6f} bits")
            print(f"directed information (previous output to next output, given input): {sum(dir_inf_POtoFO):.6f} bits")
            print(f"dir_inf_ItoO per token: {sum(dir_inf_ItoO) / output_token_count:.6f} bits/token")
            print(f"dir_inf_POtoFO per token: {sum(dir_inf_POtoFO) / output_token_count:.6f} bits/token")

    print(f"\nMaximum dir_inf_ItoO across all question trials: {max(all_dir_inf_ItoO):.6f} bits")
    print(f"Maximum dir_inf_POtoFO across all question trials: {max(all_dir_inf_POtoFO):.6f} bits")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=10, help="Number of questions, 1–8000 (default: 10).")
    parser.add_argument("--trials", type=int, default=1, help="Responses per question (positive integer; default: 1).")
    args = parser.parse_args()
    if not 1 <= args.sample <= 8000:
        parser.error("--sample must be between 1 and 8000.")
    if args.trials < 1:
        parser.error("--trials must be a positive integer.")
    main(sample=args.sample, trials=args.trials)
