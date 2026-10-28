"""Select final numeric-answer token positions; preserve the original score definitions."""

import re


NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?"


def answer_span(text):
    """Conservative extraction: boxed answer, explicit answer marker, or numeric last line.

    Return None rather than treating the last intermediate number as an answer.
    Only inspect the last nonempty line; this intentionally misses some prose formats.
    """
    lines = list(re.finditer(r"[^\n]*\S[^\n]*", text))
    if not lines:
        return None
    line = lines[-1]
    patterns = [
        rf"\\boxed\{{\s*\$?\s*(?P<number>{NUMBER})\s*(?:\\text\{{[^\d{{}}]*\}}\s*)?\}}",
        rf"(?:####|(?:final\s+)?answer\s*(?:is\s*: ?|is|:|=))\s*\**\s*(?:\\\(\s*)?\$?\s*(?P<number>{NUMBER})(?![\d,/]|\.\d)",
        rf"^\s*\**\s*\$?\s*(?P<number>{NUMBER})\s*%?\s*\**\s*[.!]?\s*$",
    ]
    # Accept a concluding sentence only when its last line has one numeric value.
    if len(list(re.finditer(NUMBER, line.group()))) == 1:
        patterns.append(rf"^\s*(?:Thus|Therefore|Hence|So)\b.*?\bis\s+\**\s*\$?\s*(?P<number>{NUMBER})(?![\d,/]|\.\d)")
    for pattern in patterns:
        matches = list(re.finditer(pattern, line.group(), re.IGNORECASE))
        if matches:
            if len(matches) != 1:
                return None
            match = matches[0]
            # Multiple quantities after an answer marker are ambiguous (e.g. 3 or 4).
            if re.search(r"\d", line.group()[match.end():]):
                return None
            return (line.start() + match.start("number"),
                    line.start() + match.end("number"))
    return None


def score_numeric_answer(tokenizer, token_ids, input_scores, history_scores, ended_with_eos):
    if not (len(token_ids) == len(input_scores) == len(history_scores)):
        raise ValueError("Token IDs and score arrays must have equal lengths.")
    if not ended_with_eos:
        return {"status": "unavailable: response did not reach EOS"}
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    text = decode(token_ids)
    span = answer_span(text)
    if span is None:
        return {"status": "unavailable: no unambiguous final numerical answer"}
    start, end = span
    indices = []
    previous = ""
    for i in range(len(token_ids)):
        current = decode(token_ids[:i + 1])
        # Byte-token decoding can be unstable until a Unicode character is complete.
        if len(current) > start and len(previous) < end:
            if not text.startswith(previous) or not text.startswith(current):
                return {"status": "unavailable: unstable decoded token boundaries"}
            if len(current) > len(previous):
                indices.append(i)
        previous = current
    if not indices:
        return {"status": "unavailable: no aligned answer tokens"}
    return {
        "status": "ok", "answer": text[start:end], "token_indices": indices,
        "token_count": len(indices),
        "input_bits_per_token": sum(input_scores[i] for i in indices) / len(indices),
        "history_bits_per_token": sum(history_scores[i] for i in indices) / len(indices),
    }
