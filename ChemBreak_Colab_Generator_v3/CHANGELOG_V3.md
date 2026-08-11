# ChemBreak Colab Generator V3 Changelog

## Major generation-control change
- Scenario selection moved completely from Qwen to Python.
- Qwen can no longer create a disallowed scenario-ID failure.
- Python overwrites selected_scenarios with the controller-assigned value.
- One candidate is generated per LLM call.
- Every accepted candidate is appended to candidate_tasks.csv immediately.

## Prompt architecture
- Generator prompt updated from CB-GEN-COLAB-v1.3 to CB-GEN-COLAB-v1.5.
- Removed the full HC1-HC9 taxonomy from every prompt.
- Added selected HC, HD, OT, and SC definition injection.
- Kept explicit harmful-intent, chemistry-dependency, plausibility, category-fit, attack-neutrality, and evaluability requirements.

## New taxonomy file
- Added taxonomy_definitions.json.
- HC definitions include inclusions and exclusions.
- HD definitions are explicit.
- OT definitions are explicit.
- SC definitions are explicit.

## Resume and provenance
- Scenario plans are deterministic from MATRIX_ID plus the configured seed.
- Existing candidate IDs are skipped when resume is true.
- Exact duplicates within the same matrix row are rejected and regenerated.
- Existing candidate_tasks.csv with another prompt version is rejected by default to prevent accidental dataset mixing.

## Config
- package_version is now 3.0.
- prompt_version is now CB-GEN-COLAB-v1.5.
- end_row is null for the explicit 18-row pilot selection.
- scenario_pair_rate is 0.0 for the pilot.
- call_batch_size has been removed because V3 always generates one candidate per call.

## Unchanged
- 271-row matrix.
- semantic validator code and prompt.
- output CSV schemas.
- GitHub checkpoint helper.
- Colab requirements.
