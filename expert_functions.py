import prompts
import expert_basics
import logging

PROB_THRESHOLD = 0.8
SCALE_THRESHOLD = 4.0

# ---------- logger ----------
if "detail_logger" not in logging.getLogger().manager.loggerDict:
    detail_logger = logging.getLogger("detail_logger")
    detail_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    detail_logger.addHandler(handler)

def log_info(msg, logger="detail_logger", print_to_std=False):
    lg = logging.getLogger(logger) if isinstance(logger, str) else logger
    if lg: lg.info(msg)
    if print_to_std: print(msg + "\n")

# ---------- helpers ----------
def answer_to_idx(answer):  # 'A'→0
    return ord(answer.upper()) - ord("A")

def build_triplet_context(patient_state):
    triplets = patient_state.get("triplets", [])
    return "None" if not triplets else "\n".join(triplets)
# -----------------------------------------------------------------


# ========================= FIXED ABSTAIN =========================
def fixed_abstention_decision(max_depth, patient_state, inquiry, options_dict, **kwargs):
    log_info("++++++++++++++++++++ Fixed Abstention ++++++++++++++++++++")
    abstain_decision = len(patient_state['interaction_history']) < max_depth
    conf_score = 1.0 if abstain_decision else 0.0

    patient_info   = patient_state["initial_info"]
    triplet_context = build_triplet_context(patient_state)
    options_text   = "A: {A}, B: {B}, C: {C}, D: {D}".format(**options_dict)

    prompt_answer  = prompts.expert_system["curr_template"].format(
        patient_info, triplet_context, inquiry, options_text, prompts.expert_system["answer"]
    )
    messages_answer = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user",   "content": prompt_answer},
    ]
    response_text, letter_choice, num_toks = expert_basics.expert_response_choice(
        messages_answer, options_dict, **kwargs
    )

    log_info(f"[FIXED] abstain={abstain_decision} conf={conf_score} choice={letter_choice}")
    return {"abstain": abstain_decision,
            "confidence": conf_score,
            "usage": num_toks,
            "messages": messages_answer,
            "letter_choice": letter_choice}


# ========================= QUESTION GEN =========================
def question_generation(patient_state, inquiry, options_dict, messages, independent_modules, **kwargs):
    task_prompt = prompts.expert_system["atomic_question_improved"]

    if independent_modules:
        patient_info    = patient_state["initial_info"]
        triplet_context = build_triplet_context(patient_state)
        options_text    = "A: {A}, B: {B}, C: {C}, D: {D}".format(**options_dict)
        prompt = prompts.expert_system["curr_template"].format(
            patient_info, triplet_context, inquiry, options_text, task_prompt
        )
        messages = [
            {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
            {"role": "user",   "content": prompt},
        ]
    else:
        messages.append({"role": "user", "content": task_prompt})

    resp_txt, atomic_q, num_toks = expert_basics.expert_response_question(messages, **kwargs)
    log_info(f"[ATOMIC QUESTION]: {atomic_q}")
    messages.append({"role": "assistant", "content": atomic_q})
    return {"atomic_question": atomic_q, "messages": messages, "usage": num_toks}


# =================================================================
# 下面所有 abstention 方案都统一改用 triplet_context
# =================================================================

def implicit_abstention_decision(patient_state, rationale_generation, inquiry, options_dict, **kwargs):
    prompt_key = "implicit_RG" if rationale_generation else "implicit"
    abstain_task_prompt = prompts.expert_system[prompt_key]

    patient_info    = patient_state["initial_info"]
    triplet_context = build_triplet_context(patient_state)
    options_text    = "A: {A}, B: {B}, C: {C}, D: {D}".format(**options_dict)

    prompt_abstain = prompts.expert_system["curr_template"].format(
        patient_info, triplet_context, inquiry, options_text, abstain_task_prompt
    )
    messages = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user",   "content": prompt_abstain},
    ]
    resp, atomic_q, letter_choice, conf, probs, nt = expert_basics.expert_response_choice_or_question(
        messages, options_dict, **kwargs
    )
    abstain_decision = atomic_q is not None or letter_choice is None

    if letter_choice is None:
        prompt_ans = prompts.expert_system["curr_template"].format(
            patient_info, triplet_context, inquiry, options_text, prompts.expert_system["answer"]
        )
        messages_ans = [
            {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
            {"role": "user",   "content": prompt_ans},
        ]
        resp2, letter_choice, nt2 = expert_basics.expert_response_choice(messages_ans, options_dict, **kwargs)
        nt["input_tokens"]  += nt2["input_tokens"]
        nt["output_tokens"] += nt2["output_tokens"]

    return {"abstain": abstain_decision, "confidence": conf, "usage": nt,
            "messages": messages, "letter_choice": letter_choice, "atomic_question": atomic_q}


def binary_abstention_decision(patient_state, rationale_generation, inquiry, options_dict, **kwargs):
    prompt_key = "binary_RG" if rationale_generation else "binary"
    abstain_task_prompt = prompts.expert_system[prompt_key]

    patient_info    = patient_state["initial_info"]
    triplet_context = build_triplet_context(patient_state)
    options_text    = "A: {A}, B: {B}, C: {C}, D: {D}".format(**options_dict)

    prompt_abstain = prompts.expert_system["curr_template"].format(
        patient_info, triplet_context, inquiry, options_text, abstain_task_prompt
    )
    messages = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user",   "content": prompt_abstain},
    ]
    resp, yn, conf, probs, nt = expert_basics.expert_response_yes_no(messages, **kwargs)
    abstain_decision = yn.lower() == "no"

    # always get intermediate answer
    prompt_ans = prompts.expert_system["curr_template"].format(
        patient_info, triplet_context, inquiry, options_text, prompts.expert_system["answer"]
    )
    messages_ans = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user",   "content": prompt_ans},
    ]
    resp2, letter_choice, nt2 = expert_basics.expert_response_choice(messages_ans, options_dict, **kwargs)
    nt["input_tokens"]  += nt2["input_tokens"]
    nt["output_tokens"] += nt2["output_tokens"]

    return {"abstain": abstain_decision, "confidence": conf, "usage": nt,
            "messages": messages, "letter_choice": letter_choice}


def numerical_abstention_decision(patient_state, rationale_generation, inquiry, options_dict, **kwargs):
    prompt_key = "numerical_RG" if rationale_generation else "numerical"
    abstain_task_prompt = prompts.expert_system[prompt_key]

    patient_info    = patient_state["initial_info"]
    triplet_context = build_triplet_context(patient_state)
    options_text    = "A: {A}, B: {B}, C: {C}, D: {D}".format(**options_dict)

    prompt_abstain = prompts.expert_system["curr_template"].format(
        patient_info, triplet_context, inquiry, options_text, abstain_task_prompt
    )
    messages = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user",   "content": prompt_abstain},
    ]
    resp, conf, probs, nt = expert_basics.expert_response_confidence_score(messages, **kwargs)

    messages.append({"role": "assistant", "content": resp})
    messages.append({"role": "user",      "content": prompts.expert_system["yes_no"]})
    resp2, yn, _, probs2, nt2 = expert_basics.expert_response_yes_no(messages, **kwargs)
    abstain_decision = yn.lower() == "no"
    nt["input_tokens"]  += nt2["input_tokens"]
    nt["output_tokens"] += nt2["output_tokens"]

    # get intermediate answer
    prompt_ans = prompts.expert_system["curr_template"].format(
        patient_info, triplet_context, inquiry, options_text, prompts.expert_system["answer"]
    )
    messages_ans = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user",   "content": prompt_ans},
    ]
    resp3, letter_choice, nt3 = expert_basics.expert_response_choice(messages_ans, options_dict, **kwargs)
    nt["input_tokens"]  += nt3["input_tokens"]
    nt["output_tokens"] += nt3["output_tokens"]

    return {"abstain": abstain_decision, "confidence": conf, "usage": nt,
            "messages": messages, "letter_choice": letter_choice}


def numcutoff_abstention_decision(patient_state, rationale_generation, inquiry, options_dict, abstain_threshold=None, **kwargs):
    if not abstain_threshold:
        abstain_threshold = PROB_THRESHOLD
    prompt_key = "numcutoff_RG" if rationale_generation else "numcutoff"
    abstain_task_prompt = prompts.expert_system[prompt_key]

    patient_info    = patient_state["initial_info"]
    triplet_context = build_triplet_context(patient_state)
    options_text    = "A: {A}, B: {B}, C: {C}, D: {D}".format(**options_dict)

    prompt_abstain = prompts.expert_system["curr_template"].format(
        patient_info, triplet_context, inquiry, options_text, abstain_task_prompt
    )
    messages = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user",   "content": prompt_abstain},
    ]
    resp, conf, probs, nt = expert_basics.expert_response_confidence_score(messages, abstain_threshold, **kwargs)
    abstain_decision = conf < abstain_threshold

    # get intermediate answer
    prompt_ans = prompts.expert_system["curr_template"].format(
        patient_info, triplet_context, inquiry, options_text, prompts.expert_system["answer"]
    )
    messages_ans = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user",   "content": prompt_ans},
    ]
    resp2, letter_choice, nt2 = expert_basics.expert_response_choice(messages_ans, options_dict, **kwargs)
    nt["input_tokens"]  += nt2["input_tokens"]
    nt["output_tokens"] += nt2["output_tokens"]

    return {"abstain": abstain_decision, "confidence": conf, "usage": nt,
            "messages": messages, "letter_choice": letter_choice}


def scale_abstention_decision(patient_state, rationale_generation, inquiry, options_dict, abstain_threshold=None, **kwargs):
    if not abstain_threshold:
        abstain_threshold = SCALE_THRESHOLD
    prompt_key = "scale_RG" if rationale_generation else "scale"
    abstain_task_prompt = prompts.expert_system[prompt_key]

    patient_info    = patient_state["initial_info"]
    triplet_context = build_triplet_context(patient_state)
    options_text    = "A: {A}, B: {B}, C: {C}, D: {D}".format(**options_dict)

    prompt_abstain = prompts.expert_system["curr_template"].format(
        patient_info, triplet_context, inquiry, options_text, abstain_task_prompt
    )
    messages = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user",   "content": prompt_abstain},
    ]

    self_consistency = kwargs.pop("self_consistency", 1)
    kwargs["self_consistency"] = self_consistency

    resp, conf, probs, nt = expert_basics.expert_response_scale_score(
        messages, **kwargs
    )
    abstain_decision = conf < abstain_threshold

    # get intermediate answer
    prompt_ans = prompts.expert_system["curr_template"].format(
        patient_info, triplet_context, inquiry, options_text, prompts.expert_system["answer"]
    )
    messages_ans = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user",   "content": prompt_ans},
    ]
    resp2, letter_choice, nt2 = expert_basics.expert_response_choice(messages_ans, options_dict, **kwargs)
    nt["input_tokens"]  += nt2["input_tokens"]
    nt["output_tokens"] += nt2["output_tokens"]

    return {
        "abstain": abstain_decision,
        "confidence": conf,
        "usage": nt,
        "messages": messages,
        "letter_choice": letter_choice
    }
