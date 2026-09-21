from __future__ import annotations
from pathlib import Path
import hashlib
import pandas as pd
from .constants import SOURCE_PROMPTS_SHA256, MANIFEST_SHA256, TASK_COUNT
from .utils import sha256_file

REQUIRED = {"Behavior", "FunctionalCategory", "SemanticCategory"}

def _behavior_hash(text: str) -> str:
    return hashlib.sha256(str(text).strip().encode()).hexdigest()

def load_prompts(path: str|Path) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Prompt dataset missing columns: {sorted(missing)}")
    if len(df) != TASK_COUNT:
        raise ValueError(f"Expected {TASK_COUNT} prompt rows, got {len(df)}")
    norm = df["Behavior"].astype(str).str.strip().str.casefold()
    if norm.duplicated().any():
        raise ValueError("Behavior values must be unique after normalization")
    if sha256_file(p) != SOURCE_PROMPTS_SHA256:
        raise ValueError("Source prompt CSV SHA-256 does not match CB26 lock")
    out = df.copy()
    out["Behavior"] = out["Behavior"].astype(str).str.strip()
    return out

def load_manifest(path: str|Path) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p)
    if len(df) != TASK_COUNT:
        raise ValueError(f"Expected {TASK_COUNT} manifest rows, got {len(df)}")
    if df["selection_order"].astype(int).tolist() != list(range(1, TASK_COUNT + 1)):
        raise ValueError("Manifest selection_order must be exactly 1..28")
    if df["assignment_id"].astype(str).duplicated().any():
        raise ValueError("Manifest assignment_id values must be unique")
    if sha256_file(p) != MANIFEST_SHA256:
        raise ValueError("CB26 manifest SHA-256 mismatch")
    return df

def selected_tasks(prompts_path: str|Path, manifest_path: str|Path) -> pd.DataFrame:
    prompts = load_prompts(prompts_path).reset_index(drop=True)
    manifest = load_manifest(manifest_path).sort_values("selection_order").reset_index(drop=True)
    rows = []
    for i, (m, src) in enumerate(zip(manifest.to_dict("records"), prompts.to_dict("records")), start=1):
        if int(m["source_row"]) != i:
            raise RuntimeError(f"Manifest/source row mismatch at position {i}")
        behavior = str(src["Behavior"]).strip()
        if _behavior_hash(behavior) != str(m["behavior_sha256"]):
            raise RuntimeError(f"Behavior hash mismatch for {m['assignment_id']}")
        rows.append({
            "selection_order": i,
            "assignment_id": str(m["assignment_id"]),
            "source_row": i,
            "behavior_sha256": str(m["behavior_sha256"]),
            "benchmark_prompt": behavior,
            "original_prompt": behavior,
            # The source file does not separately annotate goal versus intent.  CB26 therefore
            # uses one immutable source-derived goal/intent anchor rather than inventing one.
            "goal_intent_anchor": behavior,
            "requested_outcome": "source_behavior",
            "functional_category": str(src["FunctionalCategory"]),
            "semantic_category": str(src["SemanticCategory"]),
        })
    return pd.DataFrame(rows)
