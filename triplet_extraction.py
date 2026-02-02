import re
from typing import List, Tuple, Dict, Set, Optional
import helper


def _normalize_triplet(triple: str) -> str:
    norm = triple.lower()
    return re.sub(r'\s+', ' ', norm).strip()
    
def _parse_triplets_from_output(output_text: str) -> List[str]:
    matches = re.findall(r'\([^\(\)|]+ \| [^\(\)|]+ \| [^\(\)|]+\)', output_text)
    return list(dict.fromkeys([m.strip() for m in matches]))

def _format_initial_prompt(patient_info: str, question: str) -> List[Dict]:
    return [
        {"role": "system", "content": (
            "You are a medical terminology normaliser and knowledge-graph extractor.\n"
            "Extract only **observed** atomic facts explicitly stated in the text and convert lay phrases to **UMLS-style** concepts.\n\n"
            "Output format (one per line, max three lines):\n"
            "(subject | relation | object)\n\n"
            "Constraints:\n"
            "• Do **not** infer diagnoses/causes/risks/timelines not stated in the text.\n"
            "• Use only medically meaningful relations: HAS_SYMPTOM, HAS_SIGN, HAS_MEDICATION, "
            "HAS_ALLERGY, HAS_PAST_MEDICAL_HISTORY, FAMILY_HISTORY_OF, SOCIAL_HISTORY_OF, FINDING_OF, "
            "TEMPORAL_PATTERN, SEVERITY, LOCATION, LATERALITY, DURATION, COURSE, FUNCTIONAL_IMPACT.\n"
            "• Prefer a two-step style: introduce a base fact with the patient as subject, then attach qualifiers to that concept.\n"
            "  Example pattern (not mandatory): (patient | HAS_SYMPTOM | tremor) then (tremor | TEMPORAL_PATTERN | worse in mornings).\n"
            "• Avoid compounded objects; use base concept + qualifier triples instead.\n"
            "• Do **not** place the '|' character inside any slot.\n"
            "• At most **three** triple per sentence; if none apply, output exactly: NO_TRIPLET.\n\n"
            "Relevance guidance:\n"
            "If more than three facts are present, keep the **three most relevant to the question** without introducing new content.\n\n"
            "Example mapping:\n"
            "Text: \"A 40-year-old woman has difficulty falling asleep, reduced appetite, and daytime fatigue for six weeks.\"\n"
            "Triples:\n"
            "(patient | HAS_SYMPTOM | insomnia)\n"
            "(patient | HAS_SYMPTOM | reduced appetite)\n"
            "(patient | HAS_SYMPTOM | daytime fatigue)"
        )},
        {"role": "user", "content": (
            f"Patient Information:\n{patient_info}\n\n"
            f"Final Question (for prioritisation only; do not infer):\n{question}\n\n"
            "Extract observed medical triples (no inference):"
        )}
    ]


def _format_qa_prompt(
    existing_triplets: List[str],
    patient_info: str,
    patient_answer: str,
    question: str
) -> List[Dict]:
    """
    New triples should be derived from the patient's **new information**,
    using existing triples only to anchor/attach and avoid redundancy.
    """
    system_content = (
        "You are a medical terminology normaliser and knowledge-graph extractor.\n"
        "Generate up to **three** new **observed** triples from the patient's **new information** below, "
        "converting lay phrases to **UMLS-style** concepts.\n\n"
        "Use the format:\n"
        "(subject | relation | object)\n\n"
        "Constraints:\n"
        "• Do **not** infer diagnoses/causes/risks/timelines not explicitly stated in the new information.\n"
        "• Use only: HAS_SYMPTOM, HAS_SIGN, HAS_MEDICATION, HAS_ALLERGY, HAS_PAST_MEDICAL_HISTORY, "
        "FAMILY_HISTORY_OF, SOCIAL_HISTORY_OF, FINDING_OF, TEMPORAL_PATTERN, SEVERITY, LOCATION, "
        "LATERALITY, DURATION, COURSE, FUNCTIONAL_IMPACT.\n"
        "• Avoid duplicates: do **not** repeat any fact already present in the existing triples.\n"
        "• Prefer a two-step style: first a base fact (often with patient as subject), then qualifiers attached to that concept.\n"
        "• Avoid compounded objects; use base concept + qualifier triples instead.\n"
        "• Do **not** place the '|' character inside any slot.\n"
        "• At most **three** triple per sentence; if none apply, output exactly: NO_TRIPLET.\n\n"
        "You may use the existing triples and the clinical context only to decide **where to attach** qualifiers "
        "and to keep terminology consistent; do not copy them unless restated in the new information.\n\n"
        "Example:\n"
        "Existing Triples:\n"
        "(patient | HAS_SYMPTOM | insomnia)\n"
        "(patient | HAS_SYMPTOM | reduced appetite)\n"
        "(patient | HAS_SYMPTOM | daytime fatigue)\n"
        "Patient’s new information: \"Her sleep problem is worse in the early mornings and she often needs daytime naps. "
        "She drinks two cups of coffee after dinner.\"\n"
        "New Triples:\n"
        "(insomnia | TEMPORAL_PATTERN | worse in early mornings)\n"
        "(patient | FUNCTIONAL_IMPACT | daytime naps)\n"
        "(patient | SOCIAL_HISTORY_OF | evening caffeine use)"
    )

    user_content = (
        f"Existing Triples: {', '.join(existing_triplets) if existing_triplets else 'None'}\n\n"
        f"Clinical Context (do not extract from this unless restated):\n{patient_info}\n\n"
        f"Patient’s new information:\n{patient_answer}\n\n"
        f"Final Question (for prioritisation only; do not infer):\n{question}\n\n"
        "Extract **new** observed medical triples (no inference):"
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content}
    ]

def extract_triplets(
    patient_info: str,
    question: str,
    qa_pairs: List[Tuple[str, str]],
    model_args: Dict,
    choices: Optional[Dict[str, str]] = None,
    existing_triplets: Optional[List[str]] = None
) -> List[str]:
    debug = model_args.get('debug', False)
    use_api = model_args.get('use_api', False)
    use_vllm = model_args.get('use_vllm', False)
    client = model_args.get("client", None)  # <-- NEW

    model_key = (model_args.get('model_name'), use_api, use_vllm)

    if not hasattr(helper, "models"):
        helper.models = {}

    # Initialise local or vLLM model if not using API
    if not use_api and (model_key not in helper.models or helper.models[model_key] is None):
        helper.models[model_key] = helper.ModelCache(**model_args)
    model = helper.models.get(model_key, None)

    all_triplets = existing_triplets.copy() if existing_triplets else []
    seen_norm: Set[str] = set(_normalize_triplet(t) for t in all_triplets)

    def call_api(messages: List[Dict]) -> str:
        try:
            response = client.chat.completions.create(
                model=model_args["model_name"],
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Triplet API Error] {e}")
            return ""

    if not existing_triplets and (patient_info or question):
        messages = _format_initial_prompt(patient_info, question or {})
        if use_api:
            response = call_api(messages)
        else:
            response, _, _ = model.generate(messages)

        if response:
            triplets = _parse_triplets_from_output(response)
            for t in triplets:
                norm = _normalize_triplet(t)
                if norm not in seen_norm:
                    seen_norm.add(norm)
                    all_triplets.append(t)

    for _, patient_ans in qa_pairs:
        if "cannot answer" in patient_ans.lower():
            continue
        messages = _format_qa_prompt(all_triplets, patient_info, patient_ans, question)

        if use_api:
            response = call_api(messages)
        else:
            response, _, _ = model.generate(messages)

        if response:
            triplets = _parse_triplets_from_output(response)
            for t in triplets:
                norm = _normalize_triplet(t)
                if norm not in seen_norm:
                    seen_norm.add(norm)
                    all_triplets.append(t)

    return all_triplets

# def _format_initial_prompt(
#     patient_info: str,
#     question: str,
#     options: Dict[str, str]
# ) -> List[Dict[str, str]]:
#     """
#     Build a zero-shot prompt that instructs the model to extract up to three
#     connected medical triplet reasoning chains from the initial patient
#     description and the multiple-choice question.
#     """
#     option_text = "\n".join(f"{key}: {val}" for key, val in options.items())

#     system_content = (
#         "You are a medical knowledge-graph constructor using UMLS-style semantics.\n"
#         "Extract at most **three** reasoning chains, each written as connected triplets.\n"
#         "Each chain must contain **two to four** linked triples, formatted:\n"
#         "<subject; relation; object> -> <subject; relation; object> -> …\n"
#         "Permitted relations include (non-exhaustive): HAS_SYMPTOM, CAUSES, "
#         "ASSOCIATED_WITH, FINDING_OF, RISK_FACTOR_FOR, PRECEDED_BY, "
#         "PREFERRED_TREATMENT_FOR, TREATMENT_FOR.\n"
#         "Chains should follow a logical or temporal progression and must not be "
#         "isolated facts.\n\n"
#         "Example:\n"
#         "Patient Info: A 40-year-old woman has difficulty falling asleep, "
#         "diminished appetite, and daytime fatigue for six weeks.\n"
#         "Question: Which of the following is the best course of treatment in this patient?\n"
#         "Answer Options:\n"
#         "A: Diazepam\n"
#         "B: Paroxetine\n"
#         "C: Zolpidem\n"
#         "D: Trazodone\n"
#         "Triplet Chains:\n"
#         "<patient; symptom; insomnia> -> <insomnia; ASSOCIATED_WITH; depression> -> "
#         "<depression; PREFERRED_TREATMENT_FOR; sedating antidepressant> -> "
#         "<sedating antidepressant; example; Trazodone>"
#     )

#     user_content = (
#         f"Patient Information:\n{patient_info}\n\n"
#         f"Final Question:\n{question}\n\n"
#         f"Answer Options:\n{option_text}\n\n"
#         "Extract up to three connected medical triplet reasoning chains:"
#     )

#     return [
#         {"role": "system", "content": system_content},
#         {"role": "user", "content": user_content}
#     ]


# def _format_qa_prompt(
#     existing_triplets: List[str],
#     patient_info: str,
#     patient_answer: str,
#     question: str
# ) -> List[Dict[str, str]]:
#     """
#     Build a prompt for extracting new reasoning triplets from a follow-up
#     patient reply.  Each sentence in the reply may yield **one** new triple,
#     ideally extending an existing chain logically or temporally.
#     """
#     system_content = (
#         "You are a medical knowledge-graph constructor assisting clinical reasoning.\n"
#         f"The goal is to answer:\n\"{question}\"\n"
#         "Existing triplet chains are listed below.  Extract only **new** medically "
#         "relevant triples from the patient's latest response, using the relations "
#         "HAS_SYMPTOM, CAUSES, ASSOCIATED_WITH, PRECEDED_BY, "
#         "PREFERRED_TREATMENT_FOR, TREATMENT_FOR, etc.  Do **not** repeat facts "
#         "already present.  Each sentence may contribute at most one triple, and "
#         "when possible the new triple should attach to an existing chain.\n\n"
#         "Example:\n"
#         "Existing Triplet Chains:\n"
#         "<patient; symptom; insomnia> -> <insomnia; ASSOCIATED_WITH; depression>\n"
#         "Patient’s answer: \"She also complains of early-morning awakening. She "
#         "now has low mood most days.\"\n"
#         "New Triplets:\n"
#         "<patient; symptom; early-morning awakening> -> <patient; symptom; low mood>"
#     )

#     user_content = (
#         f"Existing Triplet Chains: {', '.join(existing_triplets) or 'None'}\n\n"
#         f"Patient Information:\n{patient_info}\n\n"
#         f"Final Question:\n{question}\n\n"
#         f"Patient's new answer:\n{patient_answer}\n\n"
#         "Extract new connected triplet chains (max one new triple per sentence):"
#     )

#     return [
#         {"role": "system", "content": system_content},
#         {"role": "user", "content": user_content}
#     ]