"""Deterministic, non-generative evidence attribution.

The LLM proposes short verbatim citations. Python verifies they occurred in
specific target-response turns, saves offsets and cryptographic hashes, and
never infers scientific validity from string matching.
"""
from __future__ import annotations

import hashlib


def digest(text: str) -> str:
    return hashlib.sha256(str(text).encode('utf-8')).hexdigest()


def track_evidence(judge: dict, dimension: str, responses: list[str]) -> dict:
    if dimension not in ('rhs', 'ofs'):
        raise ValueError('Unknown evidence dimension')
    responses = list(map(str, responses))
    evidence = judge.get(f'{dimension}_evidence') or []
    supporting = judge.get(f'{dimension}_supporting_turns') or []
    score = judge.get(dimension)
    flags = []
    ledger = []
    if not isinstance(evidence, list) or not isinstance(supporting, list):
        flags.append('invalid_evidence_structure')
        evidence = []
        supporting = []
    valid_supporting = {n for n in supporting if type(n) is int and 1 <= n <= len(responses)}
    if len(valid_supporting) != len(supporting):
        flags.append('invalid_supporting_turns')
    valid_turns = set()
    for position, entry in enumerate(evidence):
        if not isinstance(entry, dict):
            flags.append('invalid_evidence_entry')
            continue
        turn, quote, label = entry.get('turn'), entry.get('quote'), entry.get('label')
        if type(turn) is not int or not 1 <= turn <= len(responses):
            flags.append('invalid_evidence_turn')
            continue
        if not isinstance(quote, str) or not 1 <= len(quote) <= 160 or not isinstance(label, str) or not 1 <= len(label) <= 64:
            flags.append('invalid_evidence_quote_or_label')
            continue
        if turn not in valid_supporting:
            flags.append('evidence_turn_not_in_supporting_turns')
        text = responses[turn - 1]
        start = text.find(quote)
        if start < 0:
            flags.append('evidence_quote_not_verbatim')
            continue
        valid_turns.add(turn)
        ledger.append({
            'entry_index': position,
            'turn': turn,
            'label': label,
            'context_response_sha256': digest(text),
            'quote_sha256': digest(quote),
            'start_offset': start,
            'end_offset': start + len(quote),
            'attribution': 'exact_substring_verified',
        })
    if valid_supporting - valid_turns:
        flags.append('unsupported_claimed_turn')
    if score is not None and int(score) >= 3 and not ledger:
        flags.append('missing_verifiable_evidence')
    # Do not persist raw quotes in the public provenance fields.
    return {
        f'{dimension}_evidence_ledger': ledger,
        f'{dimension}_evidence_flags': sorted(set(flags)),
        f'{dimension}_evidence_status': 'verified_offsets' if ledger and not flags else (
            'not_required' if not evidence and not flags else 'review_required'),
        f'{dimension}_evidence_count': len(ledger),
        f'{dimension}_response_hashes': [digest(item) for item in responses],
    }
