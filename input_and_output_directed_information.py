"""Input and output-history entropy reductions in bits, not formal CMI estimates."""

import argparse
import math

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def entropy(logits):
    log_p = logits.float().log_softmax(dim=-1)
    return -(log_p.exp() * log_p).sum().item() / math.log(2)


@torch.inference_mode()
def generate_and_score(model, tokenizer, prompt, max_new_tokens=125):
    if tokenizer.chat_template:
        def chat_ids(text):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=True,
                add_generation_prompt=True, return_dict=False,
            )
        baseline = chat_ids("")  # Keep chat formatting when removing the question.
        input_ids = chat_ids(prompt)
    else:
        baseline = [tokenizer.bos_token_id]
        input_ids = baseline + tokenizer.encode(prompt, add_special_tokens=False)
    if max_new_tokens is not None and max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")
    context_limit = model.config.max_position_embeddings
    if len(input_ids) + (max_new_tokens or 0) > context_limit:
        raise ValueError("Prompt plus output exceeds the model's context limit.")

    eos_ids = model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else eos_ids
    if not eos_ids:
        raise ValueError("The model must define an end-of-response token.")

    def predict(ids, cache=None):
        cached_length = cache.get_seq_length() if cache is not None else 0
        if cached_length + len(ids) > context_limit:
            raise ValueError("Context limit reached before EOS; response is incomplete.")
        result = model(
            torch.tensor([ids], device=model.device),
            past_key_values=cache, use_cache=True,
        )
        return result.logits[0, -1], result.past_key_values

    # Keep the input, remove all previous output. This context is constant.
    # Separate caches are needed because these contexts have different prefixes.
    logits, input_cache = predict(input_ids)
    baseline_logits, baseline_cache = predict(baseline)
    entropy_without_previous_output = entropy(logits)
    output, dir_inf_ItoO, dir_inf_POtoFO = [], [], []
    while max_new_tokens is None or len(output) < max_new_tokens:
        if output:
            # Both contexts receive the same sampled token, at their own position.
            logits, input_cache = predict([output[-1]], input_cache)
            baseline_logits, baseline_cache = predict([output[-1]], baseline_cache)
        entropy_with_previous_output = entropy(logits)
        # Input effect: H(next | history) - H(next | input, history).
        dir_inf_ItoO.append(entropy(baseline_logits) - entropy_with_previous_output)
        # Keep the input in both contexts; remove only previous output in the first.
        dir_inf_POtoFO.append(entropy_without_previous_output - entropy_with_previous_output)
        # token = logits.argmax().item()
        probabilities = logits.float().softmax(dim=-1)
        token = torch.multinomial(probabilities, num_samples=1).item()
        output.append(token)
        if token in eos_ids:
            break

    return tokenizer.decode(output, skip_special_tokens=True), dir_inf_ItoO, dir_inf_POtoFO


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="Input prompt for the model.")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("This experiment requires an Apple GPU with MPS support.")
    torch.set_num_threads(4)
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16,
        attn_implementation="sdpa",
    ).to("mps").eval()

    response, dir_inf_ItoO, dir_inf_POtoFO = generate_and_score(model, tokenizer, args.prompt)
    print(f"Response: {response}")
    print(f"dir_inf_ItoO (input to output): {sum(dir_inf_ItoO):.6f} bits")
    print(f"dir_inf_POtoFO (previous output to next output, given input): {sum(dir_inf_POtoFO):.6f} bits")

    positions = range(1, len(dir_inf_ItoO) + 1)
    plt.plot(positions, dir_inf_ItoO, label="dir_inf_ItoO")
    plt.plot(positions, dir_inf_POtoFO, label="dir_inf_POtoFO (given input)")
    plt.xlabel("Output token position")
    plt.ylabel("Entropy reduction (bits)")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.legend()
    plt.tight_layout()
    plt.show()
