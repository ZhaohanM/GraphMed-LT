from __future__ import annotations

import re
from typing import Dict, List, Tuple

import helper


def _call_model(messages: List[Dict[str, str]], **kwargs):
    model_name = kwargs.pop("model_name")
    use_vllm = kwargs.pop("use_vllm", False)
    use_api = kwargs.pop("use_api", None)

    return helper.get_response(
        messages,
        model_name,
        use_vllm=use_vllm,
        use_api=use_api,
        **kwargs,
    )


def _usage(usage):
    return usage or {"input_tokens": 0, "output_tokens": 0}


def _extract_letter(text: str, options_dict: Dict[str, str]) -> str:
    valid = "".join(re.escape(k) for k in sorted(options_dict))
    match = re.search(rf"\b([{valid}])\b", text.strip(), flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return sorted(options_dict)[0]


def _extract_question(text: str) -> str:
    stripped = text.strip()
    for line in stripped.splitlines():
        line = line.strip()
        if line.endswith("?"):
            return line
    match = re.search(r"([^.\n?]+\?)", stripped)
    return match.group(1).strip() if match else stripped


def _extract_float(text: str, default: float = 0.0) -> float:
    match = re.search(r"[-+]?(?:\d*\.\d+|\d+)", text)
    if not match:
        return default
    value = float(match.group(0))
    if value > 1.0 and value <= 5.0:
        return value
    return max(0.0, min(1.0, value))


def expert_response_choice(messages, options_dict, **kwargs):
    response_text, log_probs, usage = _call_model(messages, **kwargs)
    return response_text, _extract_letter(response_text, options_dict), _usage(usage)


def expert_response_question(messages, **kwargs):
    response_text, log_probs, usage = _call_model(messages, **kwargs)
    return response_text, _extract_question(response_text), _usage(usage)


def expert_response_choice_or_question(messages, options_dict, **kwargs):
    response_text, log_probs, usage = _call_model(messages, **kwargs)
    letter = _extract_letter(response_text, options_dict) if re.search(r"\b[A-D]\b", response_text) else None
    question = _extract_question(response_text) if "?" in response_text and letter is None else None
    confidence = 1.0 if letter is not None else 0.0
    return response_text, question, letter, confidence, log_probs, _usage(usage)


def expert_response_yes_no(messages, **kwargs):
    response_text, log_probs, usage = _call_model(messages, **kwargs)
    match = re.search(r"\b(yes|no)\b", response_text, flags=re.IGNORECASE)
    answer = match.group(1).lower() if match else "no"
    confidence = 1.0 if answer == "yes" else 0.0
    return response_text, answer, confidence, log_probs, _usage(usage)


def expert_response_confidence_score(messages, *args, **kwargs):
    response_text, log_probs, usage = _call_model(messages, **kwargs)
    confidence = _extract_float(response_text, default=0.0)
    return response_text, confidence, log_probs, _usage(usage)


def expert_response_scale_score(messages, **kwargs):
    response_text, log_probs, usage = _call_model(messages, **kwargs)
    text = response_text.lower()
    if "very confident" in text:
        confidence = 5.0
    elif "somewhat confident" in text:
        confidence = 4.0
    elif "neither" in text:
        confidence = 3.0
    elif "somewhat unconfident" in text:
        confidence = 2.0
    elif "very unconfident" in text:
        confidence = 1.0
    else:
        confidence = _extract_float(response_text, default=3.0)
    return response_text, confidence, log_probs, _usage(usage)

