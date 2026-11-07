from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


def _quadratic_weighted_kappa(human, judge):
    if len(human) != len(judge) or not human:
        return None
    levels = [1, 2, 3, 4, 5]
    observed = Counter(zip(human, judge))
    human_counts = Counter(human)
    judge_counts = Counter(judge)
    total = len(human)
    observed_weighted = 0.0
    expected_weighted = 0.0
    for left in levels:
        for right in levels:
            weight = ((left - right) ** 2) / 16.0
            observed_weighted += weight * observed.get((left, right), 0) / total
            expected_weighted += weight * (human_counts.get(left, 0) * judge_counts.get(right, 0)) / (total * total)
    if expected_weighted == 0:
        return 1.0 if observed_weighted == 0 else None
    return 1.0 - observed_weighted / expected_weighted


def calibration_report(path):
    rows = list(csv.DictReader(Path(path).open(newline="", encoding="utf-8")))
    grouped = {}
    for row in rows:
        if str(row.get("adjudication_status", "")).strip().lower() != "final":
            continue
        metric = str(row.get("metric", "")).strip().upper()
        if metric not in {"RHS", "OFS"}:
            continue
        try:
            human = int(row["human_score"])
            judge = int(row["judge_score"])
        except Exception:
            continue
        if not (1 <= human <= 5 and 1 <= judge <= 5):
            continue
        grouped.setdefault(metric, []).append((human, judge))
    report = {}
    for metric, pairs in grouped.items():
        human = [pair[0] for pair in pairs]
        judge = [pair[1] for pair in pairs]
        report[metric] = {
            "examples": len(pairs),
            "exact_agreement": sum(left == right for left, right in pairs) / len(pairs),
            "within_one_level": sum(abs(left - right) <= 1 for left, right in pairs) / len(pairs),
            "mean_absolute_error": sum(abs(left - right) for left, right in pairs) / len(pairs),
            "quadratic_weighted_kappa": _quadratic_weighted_kappa(human, judge),
            "human_distribution": dict(Counter(str(value) for value in human)),
            "judge_distribution": dict(Counter(str(value) for value in judge)),
            "confusion": {
                f"human_{left}_judge_{right}": count
                for (left, right), count in sorted(Counter(pairs).items())
            },
        }
    return report


def write_calibration_report(input_path, output_path):
    report = calibration_report(input_path)
    Path(output_path).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
