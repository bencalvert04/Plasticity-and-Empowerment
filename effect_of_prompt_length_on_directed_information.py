import math

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
def generate_and_score(prompt_ids, max_new_tokens=125):
    bos = [tokenizer.bos_token_id]
    if len(bos + prompt_ids) + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Prompt plus output exceeds the model's context limit.")
    output = []
    total = 0.0

    for _ in range(max_new_tokens):
        # Both distributions predict the next token, before it is appended.
        logits_with_prompt = model(torch.tensor([bos + prompt_ids + output]), use_cache=False).logits[0, -1]
        logits_without_prompt = model(torch.tensor([bos + output]), use_cache=False).logits[0, -1]
        entropy_change = entropy(logits_without_prompt) - entropy(logits_with_prompt)
        total += entropy_change


        # Greedy alternative: replace the next two lines with this line.
        token = logits_with_prompt.argmax().item()
        # probabilities = logits_with_prompt.softmax(dim=-1)
        # token = torch.multinomial(probabilities, num_samples=1).item()
        output.append(token)  # Both contexts receive the same generated token.
        if token == tokenizer.eos_token_id:
            break

    return total


if __name__ == "__main__":
    prompts = [
        "ben is a great person who always helps his friends when they are struggling and takes time to listen before offering advice. Last weekend he spent the morning helping a neighbor repair a broken fence and then brought dinner to a friend who had been sick. He believes that small acts of kindness can make an ordinary day feel much better for everyone around him.",
        "ray and claire sitting in a tree watched the sunset over the river while they talked about their plans for the summer. They hoped to visit the mountains, explore a few quiet towns, and spend several nights camping beneath the stars. As the evening grew darker, they climbed down carefully and followed the winding path back toward the lights of their neighborhood.",
        "you smell pretty funny after spending the entire afternoon working in the garden and carrying bags of soil through the rain. Your boots are covered in mud, your jacket is soaked, and there are leaves caught in your hair. Before sitting down for dinner, you decide to leave your clothes by the door and take a warm shower to wash away the dirt.",
        "his hair was down to his knees and moved gently in the wind as he walked along the narrow path beside the lake. Across the water, a small boat drifted near the reeds while birds gathered on the wooden dock. He stopped beneath an old willow tree, opened his notebook, and began describing the peaceful scene before the afternoon light faded.",
        "soda is bad for you when you drink too much of it every day instead of choosing water with your meals and snacks. At the grocery store, you decide to buy a large bottle of water and some fresh fruit for the week. You also plan to prepare simple lunches at home so that choosing a different drink becomes part of your daily routine.",
    ]
    tokenized_prompts = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
    if any(len(ids) < 50 for ids in tokenized_prompts):
        raise ValueError("Every prompt must contain at least 50 tokens.")
    import matplotlib.pyplot as plt; average_scores = []
    for length in range(5, 51, 2):
        scores = [generate_and_score(ids[:length]) for ids in tokenized_prompts]
        average_score = sum(scores) / len(scores)
        average_scores.append(average_score)
        print(f"Input length {length} tokens: mean sum of entropy changes = {average_score:.6f} bits")
    plt.plot(range(5, 51, 2), average_scores, "o-"); plt.xlabel("Input token length"); plt.ylabel("Mean sum of entropy changes (bits)"); plt.tight_layout(); plt.show()
