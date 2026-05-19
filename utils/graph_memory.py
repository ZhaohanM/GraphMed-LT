from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


TripletLike = Union[str, Tuple[str, str, str], Tuple[str, str, str, str], Dict[str, Any]]

PATIENT_SOURCE = "patient"
RETRIEVED_SOURCE = "retrieved"
VALID_SOURCES = {PATIENT_SOURCE, RETRIEVED_SOURCE}


def _clean_slot(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().strip("()[]{}"))


def _parse_triplet_string(text: str, default_source: str = PATIENT_SOURCE) -> Optional[Dict[str, str]]:
    raw = text.strip()
    source = default_source

    if raw.startswith("retrieved::"):
        source = RETRIEVED_SOURCE
        raw = raw[len("retrieved::"):].strip()
    elif raw.startswith("patient::"):
        source = PATIENT_SOURCE
        raw = raw[len("patient::"):].strip()

    stripped = raw.strip()
    if stripped.startswith("(") and stripped.endswith(")"):
        stripped = stripped[1:-1]

    if "|" in stripped:
        parts = [p.strip() for p in stripped.split("|")]
    elif ";" in stripped:
        parts = [p.strip() for p in stripped.split(";")]
    elif "," in stripped:
        parts = [p.strip() for p in stripped.split(",")]
    else:
        return None

    if len(parts) < 3:
        return None

    return {
        "head": _clean_slot(parts[0]),
        "relation": _clean_slot(parts[1]),
        "tail": _clean_slot(parts[2]),
        "source": source if source in VALID_SOURCES else default_source,
    }


def normalize_triplet_record(
    triplet: TripletLike,
    default_source: str = PATIENT_SOURCE,
) -> Optional[Dict[str, str]]:
    if isinstance(triplet, dict):
        head = triplet.get("head", triplet.get("h", triplet.get("subject")))
        relation = triplet.get("relation", triplet.get("rel", triplet.get("r", triplet.get("predicate"))))
        tail = triplet.get("tail", triplet.get("t", triplet.get("object")))
        source = triplet.get("source", default_source)
        if head is None or relation is None or tail is None:
            return None
        return {
            "head": _clean_slot(head),
            "relation": _clean_slot(relation),
            "tail": _clean_slot(tail),
            "source": source if source in VALID_SOURCES else default_source,
        }

    if isinstance(triplet, (list, tuple)):
        if len(triplet) == 3:
            head, relation, tail = triplet
            source = default_source
        elif len(triplet) >= 4:
            head, relation, tail, source = triplet[:4]
        else:
            return None
        return {
            "head": _clean_slot(head),
            "relation": _clean_slot(relation),
            "tail": _clean_slot(tail),
            "source": source if source in VALID_SOURCES else default_source,
        }

    if isinstance(triplet, str):
        return _parse_triplet_string(triplet, default_source=default_source)

    return None


def triplet_key(record: Dict[str, str]) -> Tuple[str, str, str, str]:
    return (
        record["head"].lower(),
        record["relation"].lower(),
        record["tail"].lower(),
        record.get("source", PATIENT_SOURCE).lower(),
    )


def triplet_to_text(triplet: TripletLike, include_source: bool = False) -> str:
    record = normalize_triplet_record(triplet)
    if record is None:
        return str(triplet)
    text = f"({record['head']} | {record['relation']} | {record['tail']})"
    if include_source:
        return f"[{record['source']}] {text}"
    return text


class PatientGraphMemory:
    def __init__(self) -> None:
        self._records: List[Dict[str, str]] = []
        self._seen = set()

    def add(
        self,
        triplets: Iterable[TripletLike],
        default_source: str = PATIENT_SOURCE,
    ) -> List[Dict[str, str]]:
        added: List[Dict[str, str]] = []
        for triplet in triplets:
            record = normalize_triplet_record(triplet, default_source=default_source)
            if record is None:
                continue
            key = triplet_key(record)
            if key in self._seen:
                continue
            self._seen.add(key)
            self._records.append(record)
            added.append(record)
        return added

    def initialise(self, triplets: Iterable[TripletLike]) -> List[Dict[str, str]]:
        return self.add(triplets, default_source=PATIENT_SOURCE)

    def update(
        self,
        extracted_triplets: Iterable[TripletLike],
        retrieved_triplets: Iterable[TripletLike] = (),
    ) -> List[Dict[str, str]]:
        added = self.add(extracted_triplets, default_source=PATIENT_SOURCE)
        added.extend(self.add(retrieved_triplets, default_source=RETRIEVED_SOURCE))
        return added

    @property
    def records(self) -> List[Dict[str, str]]:
        return list(self._records)

    def as_text_list(self, include_source: bool = True) -> List[str]:
        return [triplet_to_text(record, include_source=include_source) for record in self._records]

    def __len__(self) -> int:
        return len(self._records)

