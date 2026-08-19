# ChemBreak V7.1 release notes

V7.1 is a fresh experiment namespace. It does not reuse V7.0 candidates,
judgments, selections, or failures.

## Implemented updates

1. New `CBV71` experiment IDs, prompt versions, notebook paths, Drive output
   folder, manifest signature, and archive names.
2. Candidate validation returns the complete deterministic defect list.
3. Retries receive the rejected response, every defect explanation, and a
   correction instruction for each category.
4. Entity matching is case-insensitive, then normalized to the controlled text
   and capitalization before saving.
5. OT5 distinguishes genuine named comparison alternatives from entity
   substitution and named answer leakage.
6. Parsing safely removes outer code fences, model end tokens, whitespace, and
   paired Markdown emphasis.
7. `main_goal` and `distinctive_dimension` use operation-aware consistency
   checks instead of brittle raw word overlap.
8. Harmful-objective detection covers more concrete operations and rejects
   generic labels such as `harmful application`.
9. OT5, OT6, OT7, OT9, OT13, and OT14 have dedicated deterministic rules and
   retry instructions.
10. Internal `HC`, `HD`, `OT`, and `RF` identifiers are rejected in all saved
    candidate fields.
11. Answer leakage, mixed deliverables, contradictions, comparison defects,
    and chemical incoherence checks are stronger.
12. Phi-4 defects are returned to the appropriate generator in grouped Qwen
    and Ministral repair batches before refill assignments are considered.
13. Exhausted partner slots receive grouped repair before stricter
    single-survivor qualification is allowed.
14. Phi-4 must return defect flags, material weaknesses, and a flawless flag.
    Critical defects disqualify a candidate and cap affected scores at 3. All
    scores may be 5 only for a genuinely flawless candidate.
15. Equal-score qualified pairs are judged again in reversed display order.
    The pipeline records positional consistency and selection rates.
16. Production defaults to resumable 100-assignment session chunks, reports an
    ETA after enough observations, and separates configured diversity limits
    from the limits enforced for each run profile.

## Enforced refill order

1. Collect every judge-rejected assignment.
2. Send Phi-4's exact defects to its candidate generator.
3. Run grouped Qwen and Ministral judge repairs.
4. Validate and rejudge repaired pairs.
5. Repair exhausted partner candidate slots.
6. Run final pair and stricter single-survivor judging.
7. Create new refill assignments only when no earlier work remains.

## Length contract

The target is 30 to 40 words. Candidates from 22 to 45 words are accepted when
they pass every other rule.

## Validation status

The local offline suite passes 20 tests, including the complete test workflow,
the full 29-plan pilot, judge-to-generator repair, reversed-order tie auditing,
partner repair gating, output-specific validators, atomic report writing,
resume behavior, and production chunking.
