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
from utils.triplets_to_graph import init_bica, make_text_mappers, triplets_to_graph
from utils.graph_memory import PatientGraphMemory
from utils.triplet_retrieval import TripletRetriever

# === your triplet extractor ===
import triplet_extraction

from openai import OpenAI

client_kwargs = {"api_key": os.environ.get("OPENAI_API_KEY", "")}
if os.environ.get("OPENAI_BASE_URL"):
    client_kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
client = OpenAI(**client_kwargs)

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
    and provides a method to turn patient-specific graph memory into
    graph-conditioned evidence tokens.
    """
    def __init__(
        self,
        ckpt_path: str,
        gnn_in_dim: int = 256,
        gnn_hidden_dim: int = 256,
        prefix_len: int = 20,
        device: Optional[torch.device] = None,
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.ckpt_path = ckpt_path
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Projection checkpoint not found at: {ckpt_path}")
        self.state = torch.load(ckpt_path, map_location=self.device)
        cfg = self.state.get("config", {})
        self.gnn_in_dim = int(cfg.get("gnn_in_dim", gnn_in_dim))
        self.gnn_hidden_dim = int(cfg.get("gnn_hidden_dim", gnn_hidden_dim))
        self.prefix_len = int(cfg.get("prefix_len", prefix_len))
        self.hidden_size = int(self.state.get("hidden_size", cfg.get("hidden_size", self.gnn_hidden_dim)))
        self.gnn_model = cfg.get("gnn_model", "gat")
        self.gnn_layers = int(cfg.get("gnn_layers", 2))
        self.gat_heads = int(cfg.get("gat_heads", 4))

        # BiCA encoder (frozen) and text mappers (trainable; weights will be loaded)
        self.bica_model, self.bica_tokenizer, self.bica_device, self.bica_dim = init_bica()
        self.node_in, self.edge_in, self.source_in = make_text_mappers(
            embedding_dim=self.bica_dim,
            gnn_in_dim=self.gnn_in_dim,
            device=self.device,
            include_source=True,
        )

        self.graph_encoder = load_gnn_model[self.gnn_model](
            in_channels=self.gnn_in_dim,
            hidden_channels=self.gnn_hidden_dim,
            out_channels=self.gnn_hidden_dim,
            num_layers=self.gnn_layers,
            dropout=0.1,
            num_heads=self.gat_heads,
        ).to(self.device)

        self.projector = TripletProjector(
            graph_encoder=self.graph_encoder,
            gnn_hidden_dim=self.gnn_hidden_dim,
            prefix_len=self.prefix_len,
            hidden_size=self.hidden_size,
        ).to(self.device)

        self._load_ckpt()
        self.projector.eval()
        self.graph_encoder.eval()
        self.node_in.eval()
        self.edge_in.eval()
        self.source_in.eval()

    def _load_ckpt(self):
        self.projector.load_state_dict(self.state["projector"])
        self.graph_encoder.load_state_dict(self.state["graph_encoder"])
        self.node_in.load_state_dict(self.state["node_in"])
        self.edge_in.load_state_dict(self.state["edge_in"])
        if "source_in" in self.state:
            self.source_in.load_state_dict(self.state["source_in"])
        logging.info(f"[ProjectionEngine] Loaded ckpt from {self.ckpt_path} with config: {self.state.get('config', {})}")

    @torch.no_grad()
    def triplets_to_prefix(self, triplets: List[Tuple[str, str, str]]) -> torch.Tensor:
        """
        Converts triplets → PyG graph → evidence tokens with shape (1, m, hidden).
        """
        if not triplets:
            return torch.empty(0, device=self.device)

        graph_data = triplets_to_graph(
            triplets=triplets,
            encoder_model=self.bica_model,
            encoder_tokenizer=self.bica_tokenizer,
            encoder_device=self.bica_device,
            node_in=self.node_in,
            edge_in=self.edge_in,
            source_in=self.source_in,
            gnn_in_dim=self.gnn_in_dim,
            device=self.device,
        )
        prefix_emb = self.projector(graph_data)  # (1, m, hidden)
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

    # Build graph evidence engine and load graphmed_lt.ckpt.
    projection_ckpt = getattr(_args, "projection_ckpt", None) or getattr(_args, "proj_ckpt", None)
    if projection_ckpt in ("", "none", "None"):
        projection_ckpt = None
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
        print("[INFO] No projection checkpoint provided; running without graph evidence tokens.")

    retriever = None
    if getattr(_args, "triplet_corpus", None):
        device = projection_engine.device if projection_engine is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if projection_engine is not None:
            bica_model = projection_engine.bica_model
            bica_tokenizer = projection_engine.bica_tokenizer
            bica_device = projection_engine.bica_device
        else:
            bica_model, bica_tokenizer, bica_device, _ = init_bica()
        retriever = TripletRetriever(
            _args.triplet_corpus,
            encoder_model=bica_model,
            encoder_tokenizer=bica_tokenizer,
            encoder_device=bica_device,
            device=device,
            max_triplets=getattr(_args, "max_corpus_triplets", None),
        )
        print(f"[INFO] Loaded {len(retriever.records)} external triplets from: {_args.triplet_corpus}")

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
            projection_engine=projection_engine,
            retriever=retriever,
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
    projection_engine: Optional[TripletProjectionEngine] = None,
    retriever: Optional[TripletRetriever] = None,
):
    """
    Multi-turn conversation loop. Each turn we extract / update triplets and,
    if a projection_engine is provided, compute graph-conditioned evidence tokens
    from the source-aware graph memory. The tensor is attached to
    patient_state["triplet_prefix"] so local expert implementations can
    refine and consume embedding-level context.
    """
    # 0) Build Expert & Patient
    expert_system = expert_class(args, sample["question"], sample["options"])
    patient_system = patient_class(args, sample)

    qa_pairs: List[Tuple[str, str]] = []
    graph_memory = PatientGraphMemory()

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
        "client": client,
        "debug": False
    }

    init_triplets = triplet_extraction.extract_triplets(
        patient_info=initial_patient_info,
        question=question_str,
        qa_pairs=[],  # none yet
        choices=sample["options"],
        model_args=triplet_model_args
    )
    retrieved_triplets = retriever.retrieve(
        initial_patient_info,
        top_k=getattr(args, "retrieval_top_k", 3),
    ) if retriever is not None else []
    graph_memory.initialise(init_triplets)
    graph_memory.update([], retrieved_triplets)

    # Seed patient_state with graph memory and optional evidence tokens.
    patient_state = patient_system.get_state()
    patient_state["triplets"] = graph_memory.records
    if projection_engine is not None:
        prefix = projection_engine.triplets_to_prefix(graph_memory.records)
        patient_state["triplet_prefix"] = prefix  # (1, m, hidden) or empty tensor

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
                existing_triplets=graph_memory.as_text_list(include_source=False),
                choices=sample["options"]
            )

            retrieved_triplets = retriever.retrieve(
                patient_response,
                top_k=getattr(args, "retrieval_top_k", 3),
            ) if retriever is not None else []
            graph_memory.update(new_qa_triplets, retrieved_triplets)

            # Update patient_state
            patient_state = patient_system.get_state()
            patient_state["triplets"] = graph_memory.records

            if projection_engine is not None:
                prefix = projection_engine.triplets_to_prefix(graph_memory.records)
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
                "triplets": graph_memory.records
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
    patient_state["triplets"] = graph_memory.records
    if projection_engine is not None:
        patient_state["triplet_prefix"] = projection_engine.triplets_to_prefix(graph_memory.records)

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
        "triplets": graph_memory.records
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
