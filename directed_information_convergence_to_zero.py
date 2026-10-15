import math
import sys
import matplotlib.pyplot as plt

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


torch.set_num_threads(4)
model_id = "EleutherAI/pythia-70m"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision="step143000")
model = AutoModelForCausalLM.from_pretrained(
    model_id, revision="step143000", dtype=torch.float64,
    attn_implementation="sdpa",
).eval()


def entropy(logits):
    log_p = logits.log_softmax(dim=-1)
    return -(log_p.exp() * log_p).sum().item() / math.log(2)


@torch.inference_mode()
def generate_and_score(prompt, max_new_tokens=125):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    bos = [tokenizer.bos_token_id]
    if len(bos + prompt_ids) + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Prompt plus output exceeds the model's context limit.")
    output = []
    total = 0.0
    y_values = []

    for _ in range(max_new_tokens):
        # Both distributions predict the next token, before it is appended.
        logits_with_prompt = model(torch.tensor([bos + prompt_ids + output]), use_cache=False).logits[0, -1]
        logits_without_prompt = model(torch.tensor([bos + output]), use_cache=False).logits[0, -1]
        entropy_change = entropy(logits_without_prompt) - entropy(logits_with_prompt)
        total += entropy_change
        y_values.append(entropy_change)


        # Greedy alternative: replace the next two lines with this line.
        token = logits_with_prompt.argmax().item()
        # probabilities = logits_with_prompt.softmax(dim=-1)
        # token = torch.multinomial(probabilities, num_samples=1).item()
        output.append(token)  # Both contexts receive the same generated token.
        if token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(output, skip_special_tokens=True), total, y_values


if __name__ == "__main__":
    import numpy as np

    prompts = [
        "ben is a great person",
        "ray and claire sitting in a tree",
        "you smell pretty funny",
        "his hair was down to his knees",
        "soda is bad for you",
        "the cat fell asleep in the sunshine",
        "our train leaves before sunrise",
        "rain was tapping against the window",
        "an old book lay open on the desk",
        "the moon disappeared behind the clouds",
        "music filled the quiet room",
        "the garden comes alive every spring",
        "i forgot my umbrella at home",
        "this coffee is too hot to drink",
        "snow covered the distant mountains",
        "the children built a castle in the sand",
        "a dog chased the ball across the park",
        "the stars look especially bright tonight",
        "the recipe calls for two cups of flour",
        "the wooden door creaked as it opened",
    ]
    all_y_values = []
    for j, prompt in enumerate(prompts, start=1):
        response, score, y_values = generate_and_score(prompt)
        print(f"Prompt {j}: {response}\nSum of entropy changes: {score:.6f} bits")
        all_y_values.append(y_values)
    # Use positions reached by every response so each mean includes every prompt.
    length = min(map(len, all_y_values))
    mean_y_values = np.mean([values[:length] for values in all_y_values], axis=0)
    plt.plot(range(1, length + 1), mean_y_values)
    plt.xlabel("Output token position")
    plt.ylabel(f"Mean entropy change across {len(prompts)} prompts (bits)")
    plt.grid(axis="y", alpha=0.3)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.show()
