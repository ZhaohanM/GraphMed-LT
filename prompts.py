expert_system = {
    "meditron_system_msg_old": "You are a medical doctor answering real-world medical entrance exam questions. Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, answer the following multiplechoice question. Base your answer on the current and standard practices referenced in medical guidelines.\nTask: You will be asked to reason through the current patient's information and either ask an information seeking question or choose an option.",

    "meditron_system_msg_original": "You are a medical doctor answering real-world medical entrance exam questions. Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, answer the following multiple choice question. Base your answer on the current and standard practices referenced in medical guidelines.",

    "meditron_system_msg": "You are a medical doctor trying to reason through a real-life clinical case. Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, respond according to the task specified by the user. Base your response on the current and standard practices referenced in medical guidelines.",

    "basic_system_msg": "You are an experienced doctor trying to make a medical decision about a patient.",
    "empty_system_msg": "",

    "only_choice": "Please answer with ONLY the correct letter choice (JUST ONE LETTER and NOTHING ELSE): A, B, C, or D.",
    "system": "You are an experienced doctor trying to make a medical decision about a patient.",

    "starter": "A patient comes into the clinic presenting with symptoms as described below:\n\nKNOWLEDGE TRIPLETS:\n",

    "question_word": "Doctor Question",
    "answer_word": "Patient Response",

    "task": "Given the knowledge triplets from above, your task is to choose one of four options that best answers the Multiple Choice Question.",

    "prompt": "Medical decision-making should prioritize questions that maximally reduce diagnostic or management uncertainty. First, review the knowledge triplets and answer options. If the triplets provide sufficient evidence to select ONE best option and reasonably exclude the others, output ONLY the correct letter choice (A, B, C, or D) and NOTHING ELSE. If key information is missing, identify the single missing detail that would BEST DISCRIMINATE among A/B/C/D by eliminating the most alternatives or confirming one mechanism. Prefer high-yield factors: mechanism of disease or therapy, time course, focal exam sign, one decisive lab/imaging datum, or a specific exposure the patient can report now. Avoid broad/multi-part/low-yield questions and do not repeat any fact already present in the triplets. If you need more information, in the first line output your confidence as a float between 0.0 and 1.0; in the second line output ONLY ONE atomic question.",

    "implicit": "Given the knowledge triplets so far, if you are confident to pick an option correctly and factually, respond with ONLY the letter choice (A, B, C, or D). Otherwise, ask ONE SPECIFIC ATOMIC QUESTION that would BEST DISCRIMINATE among A/B/C/D. Do not repeat information already in the triplets; avoid broad or multi-part questions.",

    "implicit_RG": "Given the knowledge triplets so far, if you are confident to pick an option correctly and factually, respond with: REASON: one sentence explaining why your choice is correct and the others can be excluded. ANSWER: the letter choice and NOTHING ELSE. If not confident, respond with: REASON: one sentence stating what single factor would most discriminate among A/B/C/D. QUESTION: ONE atomic, high-yield question the patient can answer now.",

    "binary": "Are you confident to pick the correct option to the Multiple Choice Question based solely on the knowledge triplets provided? Answer with YES or NO and NOTHING ELSE.",

    "binary_RG": "Based on the knowledge triplets, state whether you can select a single best option and exclude the alternatives. Respond with: REASON: one sentence explaining why you are or are not confident and which single missing factor would most discriminate among A/B/C/D. DECISION: YES or NO.",

    "numcutoff": "What is your confidence score to pick the correct option using ONLY the knowledge triplets? Answer with a float from 0.0 to 1.0 and NOTHING ELSE.",
    "numcutoff_RG": "REASON: one sentence explaining your confidence and the most discriminative missing factor if any. SCORE: a float between 0.0 and 1.0.",

    "numerical": "What is your confidence score to pick the correct option using ONLY the knowledge triplets? Answer with the probability as a float from 0.0 to 1.0 and NOTHING ELSE.",
    "numerical_RG": "REASON: one sentence explaining your confidence and the single most discriminative missing factor if any. SCORE: your confidence as a float from 0.0 to 1.0.",

    "scale": "How confident are you in choosing the correct answer based on the knowledge triplets? Choose one: Very Confident; Somewhat Confident; Neither Confident nor Unconfident; Somewhat Unconfident; Very Unconfident.",

    "scale_RG": "Identify key discriminative factors from the triplets (mechanism, time course, focal exam, one decisive lab/imaging, specific exposure) and whether they suffice to confirm one option and exclude the rest. Respond with: REASON: one sentence summarizing why you are/aren’t confident and the single most discriminative missing factor if any. DECISION: one of Very Confident; Somewhat Confident; Neither Confident nor Unconfident; Somewhat Unconfident; Very Unconfident.",

    "verbal_abstain_llama": "Up to this point, based on the knowledge triplets alone, are you confident to pick the correct option? DECISION: YES or NO.",

    "implicit_abstain": "If you are confident based on the knowledge triplets, answer with ONLY the correct letter choice. If you are not confident, ask ONE SPECIFIC ATOMIC QUESTION that would BEST DISCRIMINATE among A/B/C/D. Avoid multi-part or low-yield questions and do not repeat known facts.",

    "atomic_question": "If information in the knowledge triplets is insufficient, identify the single missing detail that would most change the choice among A/B/C/D (mechanism, time course, focal exam sign, one decisive test result, or a specific exposure). Ask ONE SPECIFIC ATOMIC QUESTION the patient can answer now. Do not repeat known facts or ask multi-part questions. Output ONLY the question.",

    "atomic_question_improved": "If the knowledge triplets are insufficient to answer confidently, identify the single most important missing detail that would BEST DISCRIMINATE among A–D by eliminating the most alternatives or confirming one mechanism. Ask ONE specific, atomic follow-up the patient can answer now. Do NOT repeat any triplet or ask multi-part/low-yield questions. Output ONLY the question.",

    "answer": "Please answer with ONLY the correct letter choice (JUST ONE LETTER and NOTHING ELSE): A, B, C, or D.",

    "non_interactive": {
        "starter": "A patient comes into the clinic presenting with findings as described in the statements below:",
        "question_prompt": "Given the information from above, your task is to choose one of four options that best answers the following question: ",
        "response": "To the best of your ability, answer with ONLY the correct letter choice and nothing else."
    },

    "curr_template": "Patient Information:\n{}\n\nKNOWLEDGE TRIPLETS:\n{}\n\nMultiple Choice Question:\n{}\n\nOptions:\n{}\n\nYOUR TASK:{}"
}


patient_system = {
    "system": "You are a truthful assistant that understands the patient's information, and you are trying to answer questions from a medical doctor about the patient. ",
    "header": "Below is a list of factual statements about the patient:\n",
    "prompt": 'Which of the above atomic factual statements answers the question? If no statement answers the question, simply say "The patient cannot answer this question, please do not ask this question again." Answer only what the question asks for. Do not provide any analysis, inference, or implications. Respond by selecting all statements that answer the question from above ONLY and NOTHING ELSE.',

    "prompt_new": """Below is a list of factual statements about the patient:\n
{}\n
Which of the above atomic factual statements answers the question? If no statement answers the question, simply say "The patient cannot answer this question, please do not ask this question again." Answer only what the question asks for. Do not provide any analysis, inference, or implications. Respond with all statements that directly answer the question from above verbatim ONLY and NOTHING ELSE, with one statement on each line.

Example:
Question from the doctor: [some question]
STATEMENTS:\n[example statement: she reports that...]\n[example statement: she has a history of...]

Question from the doctor: {}
""",

    "system_first_person": "You are a patient with a list of symptoms, and you task is to truthfully answer questions from a medical doctor. ",
    "header_first_person": "Below is a list of atomic facts about you, use ONLY the information in this list and answer the doctor's question.",
    "prompt_first_person": """Which of the above atomic factual statements are the best answer to the question? Select at most two statements. If no statement answers the question, simply say "The patient cannot answer this question, please do not ask this question again." Do not provide any analysis, inference, or implications. Respond by reciting the matching statements, then convert the selected statements into first person perspective as if you are the patient but keep the same information. Generate your answer in this format:

STATEMENTS: 
FIRST PERSON: """
}

conformal_scores = {
    "prompt_score": "Given the information from above, your task is to assign a likelihood score to each option. Respond with the probability as a float from 0 to 1 and NOTHING ELSE. Respond in the following format:\nA: 0.0\nB: 0.0\nC: 0.0\nD: 0.0",
}