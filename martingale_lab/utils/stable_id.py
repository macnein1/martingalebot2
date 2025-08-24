import hashlib
import json
from typing import Dict


def make_stable_id(payload: Dict, *, run_id: str, batch_idx: int, cand_idx: int) -> str:
    """
    Build a deterministic, candidate-specific stable_id using selected keys
    and contextual identifiers (run_id, batch_idx, cand_idx).
    """
    key = {
        "run_id": run_id,
        "batch_idx": batch_idx,
        "cand_idx": cand_idx,
        "overlap": payload.get("overlap"),
        "orders": payload.get("orders"),
        "alpha": payload.get("alpha"),
        "beta": payload.get("beta"),
        "gamma": payload.get("gamma"),
        "lambda_penalty": payload.get("lambda_penalty"),
        "wave_pattern": payload.get("wave_pattern"),
        "tail_cap": payload.get("tail_cap"),
        "min_indent_step": payload.get("min_indent_step"),
        "softmax_temp": payload.get("softmax_temp"),
        "seed": payload.get("seed"),
    }
    s = json.dumps(key, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()
