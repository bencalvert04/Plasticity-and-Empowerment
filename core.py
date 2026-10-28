"""Compare next-token distributions along ONE generated response."""
import re
import random
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ANSWER = "\nEnd with Final answer: followed by a number on its own line."


def entropy(p):
    return -torch.special.xlogy(p, p).sum() / math.log(2)


def compare(full, baseline):
    js = entropy((full + baseline) / 2) - (entropy(full) + entropy(baseline)) / 2
    return [float(entropy(baseline) - entropy(full)), max(0.0, float(js))]


class Experiment:
    def __init__(self):
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float16 if device == "mps" else torch.float32,
        ).to(device).eval()
        eos = self.model.generation_config.eos_token_id
        self.eos = {eos} if isinstance(eos, int) else set(eos)

    def chat(self, *contents):
        messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": text}
                    for i, text in enumerate(contents)]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False)

    @torch.inference_mode()
    def predict(self, ids, cache=None):
        past = cache.get_seq_length() if cache is not None else 0
        if past + len(ids) > self.model.config.max_position_embeddings:
            raise ValueError("Context limit exceeded")
        x = torch.tensor([ids], device=self.model.device)
        result = self.model(x, past_key_values=cache, use_cache=True, logits_to_keep=1)
        return result.logits[0, -1].float().softmax(-1), result.past_key_values

    def run(self, contexts, limit=125, greedy=False, history=False):
        """First context generates; other contexts receive the SAME output tokens.

        history=True adds the fixed question-only baseline used in the original study.
        Each context has its own KV cache; subsequent calls process one token.
        """
        if limit < 1:
            raise ValueError("Token limit must be positive")
        names = list(contexts)[1:] + (["history"] if history else [])
        scores = {name: [] for name in names}
        tokens = []
        states = [self.predict(ids) for ids in contexts.values()]
        initial = states[0][0]
        for _ in range(limit):
            p = states[0][0]
            others = [probabilities for probabilities, _ in states[1:]]
            if history:
                others.append(initial)
            for name, q in zip(names, others):
                scores[name].append(compare(p, q))
            token = p.argmax().item() if greedy else torch.multinomial(p, 1).item()
            tokens.append(token)
            if token in self.eos:
                break
            if len(tokens) < limit:
                states = [self.predict([token], cache) for _, cache in states]
        text = self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        done = tokens[-1] in self.eos
        answer = answer_indices(self.tokenizer, tokens, text) if done else []
        return dict(text=text, tokens=tokens, complete=done, scores=scores,
                    answer_indices=answer, answer_means={
                        name: torch.tensor(values)[answer].mean(0).tolist() if answer else None
                        for name, values in scores.items()})

    def prompt(self, text, limit=125, greedy=False):
        return self.run({"full": self.chat(text), "input": self.chat("")}, limit, greedy, history=True)

    def inject(self, question, correction, pause_max=32):
        pause = random.randint(1, pause_max)
        first = self.run({"full": self.chat(question)}, pause)
        if first["complete"]:
            return dict(pause=pause, first=first, skipped="Response ended before interruption")
        # Splice exact sampled IDs into the assistant turn, including partial byte tokens.
        def context(span, message):
            prefix = self.chat(question)
            empty = self.chat(question, "", message + ANSWER)
            assert empty[:len(prefix)] == prefix, "Expected Qwen chat template"
            return prefix + span + empty[len(prefix):]
        contexts = {"full": context(first["tokens"], correction),
                    "injection": context(first["tokens"], ""),
                    "first_span": context([], correction)}
        continuation = self.run(contexts, 10)
        count = sum(token not in self.eos for token in continuation["tokens"])
        means = {name: torch.tensor(values).mean(0).tolist() if count == 10 else None
                 for name, values in continuation["scores"].items()}
        return dict(pause=pause, first=first, continuation=continuation,
                    scored_tokens=count, window_means=means)


def answer_indices(tokenizer, tokens, text):
    match = re.search(r"Final answer:\s*([+-]?\d+(?:\.\d+)?)\s*$", text)
    if not match:
        return []
    prefixes = [tokenizer.decode(tokens[:i], skip_special_tokens=True,
                                clean_up_tokenization_spaces=False) for i in range(len(tokens) + 1)]
    if any(not text.startswith(prefix) for prefix in prefixes):
        return []
    start, end = match.span(1)
    return [i for i in range(len(tokens)) if len(prefixes[i]) < end
            and len(prefixes[i + 1]) > start and len(prefixes[i + 1]) > len(prefixes[i])]
