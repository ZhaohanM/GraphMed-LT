import json
import os
import time
import logging
import importlib
from typing import List, Tuple, Dict, Any, Optional

import torch

from args import get_args
from patient import Patient

# === projection stack (matches the training code) ===
from utils.triplet_projector import TripletProjector
from utils.gnn import load_gnn_model
from utils.triplets_to_graph import init_sbert, make_text_mappers, triplets_to_graph

# === your triplet extractor ===
import triplet_extraction

from openai import OpenAI
client = OpenAI(
    base_url="http://api.llm.apps.os.dcs.gla.ac.uk/v1",
    api_key=os.environ["IDA_LLM_API_KEY"]
)

def setup_logger(name: str, file: Optional[str]):
    if not file:
        return None
    logger = logging.getLogger(name)
    # avoid duplicate handlers if re-run in same process
    if any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(file)
           for h in logger.handlers):
        return logger
    handler = logging.FileHandler(file, mode='a', encoding='utf-8')
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def log_info(message: str, print_to_std: bool = False):
    if history_logger:
        history_logger.info(message)
    if detail_logger:
        detail_logger.info(message)
    if print_to_std:
        print(message + "\n")


def load_data(filename: str) -> Dict[str, Dict[str, Any]]:
    with open(filename, "r", encoding="utf-8") as json_file:
        data = [json.loads(line) for line in json_file]
    return {item['id']: item for item in data}


class TripletProjectionEngine:
    """
    Loads the trained projection checkpoint (projector + GNN + text mappers),
    and provides a method to turn a list of triplets into a soft-prefix embedding.

    It does NOT apply the prefix to any LLM here; instead, it returns the prefix tensor.
    Your expert module can read `patient_state["triplet_prefix"]` and inject it into
    the model call (e.g., by concatenating with token embeddings before forward).
    """
    def __init__(
        self,
        ckpt_path: str,
        gnn_in_dim: int = 768,
        gnn_hidden_dim: int = 768,
        prefix_len: int = 20,
        device: Optional[torch.device] = None,
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.ckpt_path = ckpt_path
        self.gnn_in_dim = gnn_in_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.prefix_len = prefix_len

        # SBERT encoder (frozen) and text mappers (trainable; weights will be loaded)
        self.sbert_model, self.sbert_tokenizer, self.sbert_device, self.sbert_dim = init_sbert()
        # mappers map SBERT embeddings to GNN input dim
        self.node_in, self.edge_in = make_text_mappers(
            sbert_dim=self.sbert_dim,
            gnn_in_dim=self.gnn_in_dim,
            device=self.device,
        )

        # GNN encoder (trainable)
        self.graph_encoder = load_gnn_model["gcn"](
            in_channels=self.gnn_in_dim,
            hidden_channels=self.gnn_hidden_dim,
            out_channels=self.gnn_hidden_dim,
            num_layers=2,
            dropout=0.1,
        ).to(self.device)

        # Projector: (graph → prefix)
        self.projector = TripletProjector(
            graph_encoder=self.graph_encoder,
            gnn_hidden_dim=self.gnn_hidden_dim,
            prefix_len=self.prefix_len,
            # NOTE: hidden_size of expert model is not strictly needed to *compute* the prefix;
            # but the projector was trained with a particular hidden size. We keep its learned
            # output layer as saved in the checkpoint.
            hidden_size=self.gnn_hidden_dim  # dummy placeholder; overwritten by state_dict
        ).to(self.device)

        self._load_ckpt(ckpt_path)
        self.projector.eval()
        self.graph_encoder.eval()
        self.node_in.eval()
        self.edge_in.eval()

    def _load_ckpt(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Projection checkpoint not found at: {path}")
        state = torch.load(path, map_location=self.device)
        # expected keys: projector, graph_encoder, node_in, edge_in
        self.projector.load_state_dict(state["projector"])
        self.graph_encoder.load_state_dict(state["graph_encoder"])
        self.node_in.load_state_dict(state["node_in"])
        self.edge_in.load_state_dict(state["edge_in"])

        # optionally read config:
        cfg = state.get("config", {})
        # sanity log (does not hard-fail if mismatch; you may enforce if desired)
        logging.info(f"[ProjectionEngine] Loaded ckpt from {path} with config: {cfg}")

    @torch.no_grad()
    def triplets_to_prefix(self, triplets: List[Tuple[str, str, str]]) -> torch.Tensor:
        """
        Converts triplets → PyG graph → projector → (1, prefix_len, hidden) tensor (device-local).
        """
        if not triplets:
            # Return an empty prefix to signal "no soft prompt"
            return torch.empty(0, device=self.device)

        graph_data = triplets_to_graph(
            triplets=triplets,
            sbert_model=self.sbert_model,
            sbert_tokenizer=self.sbert_tokenizer,
            sbert_device=self.sbert_device,
            node_in=self.node_in,
            edge_in=self.edge_in,
            gnn_in_dim=self.gnn_in_dim,
            device=self.device,
        )
        prefix_emb = self.projector(graph_data)  # (1, prefix_len, hidden)
        return prefix_emb

def main():
    # Parse args
    _args = get_args()

    # Device info
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"[INFO] CUDA is available. Using GPU: {device_name}")
    else:
        print("[INFO] CUDA is NOT available. Using CPU.")

    # Prepare logs
    patient_data_path = os.path.join(_args.data_dir, _args.dev_filename)
    _patient_data = load_data(patient_data_path)

    # Load already processed output (to skip)
    processed_ids = _load_processed_ids(_args.output_filename)

    # Load modules
    expert_module = importlib.import_module(_args.expert_module)
    expert_class = getattr(expert_module, _args.expert_class)

    patient_module = importlib.import_module(_args.patient_module)
    patient_class = getattr(patient_module, _args.patient_class)

    # === NEW: Build projection engine and load projection.ckpt ===
    projection_ckpt = getattr(_args, "projection_ckpt", None) or getattr(_args, "proj_ckpt", None)
    if not projection_ckpt:
        # sensible default name consistent with training snippet
        projection_ckpt = getattr(_args, "save_ckpt", "proj_ce_only.ckpt")
    # Allow turning off projection by setting projection_ckpt="" in args
    projection_engine = None
    if projection_ckpt:
        projection_engine = TripletProjectionEngine(
            ckpt_path=projection_ckpt,
            gnn_in_dim=getattr(_args, "gnn_in_dim", 768),
            gnn_hidden_dim=getattr(_args, "gnn_hidden_dim", 768),
            prefix_len=getattr(_args, "prefix_len", 20),
        )
        print(f"[INFO] Loaded projection checkpoint from: {projection_ckpt}")
    else:
        print("[INFO] No projection checkpoint provided; running without soft-prefix projection.")

    # Iterate patients
    num_processed = 0
    correct_history, timeout_history, turn_lengths = [], [], []

    # Instantiate expert once if your implementation allows re-use (optional)
    for pid, sample in _patient_data.items():
        if pid in processed_ids:
            print(f"Skipping patient {pid} as it has already been processed.")
            _carry_stats(processed_ids[pid], correct_history, timeout_history, turn_lengths)
            continue

        log_info(f"|||||||||||||||||||| PATIENT #{pid} ||||||||||||||||||||")

        letter_choice, questions, answers, temp_choice_list, temp_additional_info, sample_info = run_patient_interaction(
            expert_class,
            patient_class,
            sample,
            args=_args,
            projection_engine=projection_engine
        )

        log_info(f"|||||||||||||||||||| Interaction ended for patient #{pid} ||||||||||||||||||||\n\n\n")

        # Build output
        output_dict = {
            "id": pid,
            "interactive_system": {
                "correct": (letter_choice == sample["answer_idx"]),
                "letter_choice": letter_choice,
                "questions": questions,
                "answers": answers,
                "num_questions": len(questions),
                "intermediate_choices": temp_choice_list,
                "temp_additional_info": temp_additional_info
            },
            "info": sample_info,
        }

        # Ensure directory exists
        out_dir = os.path.dirname(_args.output_filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(_args.output_filename, 'a+', encoding="utf-8") as f:
            f.write(json.dumps(output_dict, ensure_ascii=False) + '\n')

        # Update stats
        correct_history.append(letter_choice == sample["answer_idx"])
        timeout_history.append(len(temp_choice_list) > _args.max_questions)
        turn_lengths.append(len(temp_choice_list))

        num_processed += 1
        accuracy = sum(correct_history) / len(correct_history) if correct_history else 0.0
        timeout_rate = sum(timeout_history) / len(timeout_history) if timeout_history else 0.0
        avg_turns = sum(turn_lengths) / len(turn_lengths) if turn_lengths else 0.0

        results_logger.info(f'Processed {num_processed}/{len(_patient_data)} patients | Accuracy: {accuracy}')
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
              f"Processed {num_processed}/{len(_patient_data)} patients | "
              f"Accuracy: {accuracy:.4f} | Timeout Rate: {timeout_rate:.4f} | Avg. Turns: {avg_turns:.2f}")

    # Final print
    accuracy = sum(correct_history) / len(correct_history) if correct_history else 0.0
    timeout_rate = sum(timeout_history) / len(timeout_history) if timeout_history else 0.0
    avg_turns = sum(turn_lengths) / len(turn_lengths) if turn_lengths else 0.0
    print(f"Accuracy: {sum(correct_history)} / {len(correct_history)} = {accuracy:.4f}")
    print(f"Timeout Rate: {sum(timeout_history)} / {len(timeout_history)} = {timeout_rate:.4f}")
    print(f"Avg. Turns: {avg_turns:.2f}")


def run_patient_interaction(
    expert_class,
    patient_class,
    sample: Dict[str, Any],
    args,
    projection_engine: Optional[TripletProjectionEngine] = None
):
    """
    Multi-turn conversation loop. Each turn we extract / update triplets and,
    if a projection_engine is provided, compute a soft-prefix embedding from these triplets.
    The soft prefix is attached to patient_state["triplet_prefix"] for the expert to use.
    """
    # 0) Build Expert & Patient
    expert_system = expert_class(args, sample["question"], sample["options"])
    patient_system = patient_class(args, sample)

    qa_pairs: List[Tuple[str, str]] = []
    all_triplets: List[Tuple[str, str, str]] = []

    temp_choice_list: List[str] = []
    temp_additional_info: List[Dict[str, Any]] = []

    # 1) Initial triplets from (initial patient info + question)
    initial_patient_info = getattr(patient_system, "initial_info", "")
    question_str = sample["question"]

    triplet_model_args = {
        "model_name": getattr(args, "triplet_model", getattr(args, "expert_model", "")),
        "use_api": getattr(args, "use_api", True),
        "use_vllm": getattr(args, "use_vllm", False),
        "temperature": getattr(args, "temperature", 0.2),
        "max_tokens": getattr(args, "max_tokens", 512),
        "top_p": getattr(args, "top_p", 0.95),
        "debug": False
    }

    init_triplets = triplet_extraction.extract_triplets(
        patient_info=initial_patient_info,
        question=question_str,
        qa_pairs=[],  # none yet
        choices=sample["options"],
        model_args=triplet_model_args
    )
    all_triplets.extend(init_triplets)

    # seed patient_state with triplets and an optional soft prefix
    patient_state = patient_system.get_state()
    patient_state["triplets"] = all_triplets
    if projection_engine is not None:
        prefix = projection_engine.triplets_to_prefix(all_triplets)
        patient_state["triplet_prefix"] = prefix  # (1, prefix_len, hidden) or empty tensor

    # 2) Turns
    while len(patient_system.get_questions()) < args.max_questions:
        log_info(f"==================== Turn {len(patient_system.get_questions()) + 1} ====================")

        # Expert decides: ask question or answer
        response_dict = expert_system.respond(patient_state)
        log_info(f"[Expert System]: {response_dict}")
        temp_additional_info.append({k: v for k, v in response_dict.items()
                                     if k not in ["type", "letter_choice", "question"]})

        if response_dict["type"] == "question":
            temp_choice_list.append(response_dict.get("letter_choice", ""))  # optional trace
            doctor_q = response_dict["question"]

            # Patient answers
            patient_response = patient_system.respond(doctor_q)
            log_info(f"[Patient System]: {patient_response}")

            # Record Q/A for incremental triplet extraction
            qa_pairs.append((doctor_q, patient_response))
            new_qa_triplets = triplet_extraction.extract_triplets(
                patient_info=initial_patient_info,
                question=question_str,
                qa_pairs=[(doctor_q, patient_response)],
                model_args=triplet_model_args,
                existing_triplets=all_triplets,
                choices=sample["options"]
            )

            # Update triplets with de-duplication (preserve order)
            if new_qa_triplets:
                for t in new_qa_triplets:
                    if t not in all_triplets:
                        all_triplets.append(t)

            # Update patient_state
            patient_state = patient_system.get_state()
            patient_state["triplets"] = all_triplets

            if projection_engine is not None:
                prefix = projection_engine.triplets_to_prefix(all_triplets)
                patient_state["triplet_prefix"] = prefix

        elif response_dict["type"] == "choice":
            # Final decision
            expert_decision = response_dict["letter_choice"]
            temp_choice_list.append(expert_decision)

            sample_info = {
                "initial_info": getattr(patient_system, "initial_info", ""),
                "correct_answer": sample.get("answer"),
                "correct_answer_idx": sample.get("answer_idx"),
                "question": sample["question"],
                "options": sample["options"],
                "context": sample.get("context", ""),
                "facts": getattr(patient_system, "facts", None),
                "triplets": all_triplets
            }
            return (
                expert_decision,
                patient_system.get_questions(),
                patient_system.get_answers(),
                temp_choice_list,
                temp_additional_info,
                sample_info
            )

        else:
            raise ValueError("Invalid response type from expert_system.")

    # 3) Reached max questions → force final
    log_info(f"==================== Max Interaction Length ({args.max_questions} turns) Reached "
             f"--> Force Final Answer ====================")
    patient_state = patient_system.get_state()
    patient_state["triplets"] = all_triplets
    if projection_engine is not None:
        patient_state["triplet_prefix"] = projection_engine.triplets_to_prefix(all_triplets)

    response_dict = expert_system.respond(patient_state)
    log_info(f"[Expert System]: {response_dict}")

    stuck_response = response_dict["letter_choice"]
    temp_additional_info.append({k: v for k, v in response_dict.items() if k != "letter_choice"})

    sample_info = {
        "initial_info": getattr(patient_system, "initial_info", ""),
        "correct_answer": sample.get("answer"),
        "correct_answer_idx": sample.get("answer_idx"),
        "question": sample["question"],
        "options": sample["options"],
        "context": sample.get("context", ""),
        "facts": getattr(patient_system, "facts", None),
        "triplets": all_triplets
    }

    return (
        stuck_response,
        patient_system.get_questions(),
        patient_system.get_answers(),
        temp_choice_list + [stuck_response],
        temp_additional_info,
        sample_info
    )


def _load_processed_ids(output_filename: str):
    """
    Returns either {} or a dict mapping id -> summary stats so we can skip processed patients.
    """
    if not os.path.exists(output_filename):
        return {}
    with open(output_filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        return {}

    output_data = [json.loads(line) for line in lines]
    processed = {
        sample["id"]: {
            "correct": sample["interactive_system"]["letter_choice"] == sample["info"]["correct_answer_idx"],
            "timeout": len(sample["interactive_system"]["intermediate_choices"]) > sample["interactive_system"]["num_questions"],  # conservative
            "turns": sample["interactive_system"]["num_questions"]
        }
        for sample in output_data
    }
    return processed


def _carry_stats(stats_row, correct_history, timeout_history, turn_lengths):
    correct_history.append(stats_row["correct"])
    timeout_history.append(stats_row["timeout"])
    turn_lengths.append(stats_row["turns"])


if __name__ == "__main__":
    args = get_args()

    # device note
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(0)
        print(f"[INFO] CUDA is available. Using GPU: {dev}")
    else:
        print("[INFO] CUDA is NOT available. Using CPU.")

    # loggers (module-level for log_info)
    results_logger = setup_logger('results_logger', args.log_filename)
    history_logger = setup_logger('history_logger', args.history_log_filename)
    detail_logger = setup_logger('detail_logger', args.detail_log_filename)
    message_logger = setup_logger('message_logger', args.message_log_filename)

    # run
    main()
