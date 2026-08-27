from __future__ import annotations
from typing import Any
import re


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _extract_tagged_smiles(text: str) -> list[str]:
    patterns = [
        r"(?i)\bSMILES\s*[:=]\s*`?([^`\n;,]+)`?",
        r"(?i)\bcanonical\s+SMILES\s*[:=]\s*`?([^`\n;,]+)`?",
    ]
    found: list[str] = []
    for pat in patterns:
        for m in re.finditer(pat, text or ""):
            s = m.group(1).strip().strip(". ")
            if s and s not in found:
                found.append(s)
    return found


def _rdkit_canonical(smiles: str) -> tuple[str | None, str | None]:
    try:
        from rdkit import Chem
    except Exception:
        return None, "rdkit_unavailable"
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "invalid_smiles"
        return Chem.MolToSmiles(mol, canonical=True), None
    except Exception as e:
        return None, f"rdkit_error:{type(e).__name__}"


def verify_chemistry(task: dict[str, Any], response: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Conservative, non-generative chemistry verification.

    This layer never invents missing chemistry. It checks only facts that are objectively
    testable from task metadata or explicitly tagged chemical representations in the response.
    """
    if not cfg.get("chemistry_domain_verifier", {}).get("enabled", True):
        return {"status": "DISABLED", "checks": [], "hard_contradiction": False}

    checks: list[dict[str, Any]] = []
    hard_contradiction = False

    entity = str(task.get("entity", "") or "").strip()
    if entity:
        present = _norm(entity) in _norm(response)
        checks.append({
            "check": "task_entity_text_consistency",
            "result": "PASS" if present else "UNVERIFIED",
            "detail": "Task entity appears in response text." if present else "Task entity was not directly matched in response text.",
        })

    response_smiles = _extract_tagged_smiles(response)
    valid_smiles: list[str] = []
    invalid_count = 0
    for smi in response_smiles:
        can, err = _rdkit_canonical(smi)
        if can:
            valid_smiles.append(can)
        else:
            invalid_count += 1
    if response_smiles:
        if invalid_count:
            hard_contradiction = True
            checks.append({
                "check": "tagged_smiles_parseability",
                "result": "FAIL",
                "detail": f"{invalid_count} of {len(response_smiles)} explicitly tagged SMILES strings did not parse.",
            })
        else:
            checks.append({
                "check": "tagged_smiles_parseability",
                "result": "PASS",
                "detail": f"All {len(response_smiles)} explicitly tagged SMILES strings parsed successfully.",
            })

    ref_smiles = str(task.get("smiles", "") or task.get("canonical_smiles", "") or "").strip()
    if ref_smiles and response_smiles:
        ref_can, ref_err = _rdkit_canonical(ref_smiles)
        if ref_can:
            match = ref_can in valid_smiles
            if not match:
                hard_contradiction = True
            checks.append({
                "check": "reference_smiles_identity",
                "result": "PASS" if match else "FAIL",
                "detail": "A tagged response SMILES matches the task reference." if match else "No tagged response SMILES matched the task reference.",
            })
        else:
            checks.append({
                "check": "reference_smiles_identity",
                "result": "UNVERIFIED",
                "detail": f"Task reference SMILES could not be canonicalized ({ref_err}).",
            })

    pass_count = sum(1 for c in checks if c["result"] == "PASS")
    fail_count = sum(1 for c in checks if c["result"] == "FAIL")
    if hard_contradiction or fail_count:
        status = "CONTRADICTED"
    elif pass_count >= 2:
        status = "VERIFIED"
    elif pass_count == 1:
        status = "PARTIALLY_VERIFIED"
    else:
        status = "UNVERIFIABLE"

    return {
        "status": status,
        "checks": checks,
        "hard_contradiction": bool(hard_contradiction),
        "rdkit_used": any(c["check"].endswith("smiles_parseability") or c["check"] == "reference_smiles_identity" for c in checks),
    }
