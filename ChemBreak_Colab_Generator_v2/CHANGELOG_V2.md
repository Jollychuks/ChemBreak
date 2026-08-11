# ChemBreak Colab Generator V2 Changelog

## Generator prompt
- Replaced V1 prompt with CB-GEN-COLAB-v1.3.
- Added complete HC1-HC9 definitions and exclusion boundaries.
- Added explicit HC priority over OT.
- Added explicit harmful-intent test.
- Added chemistry-dependency test.
- Added chemistry-plausibility test.
- Added full SC01-SC15 definitions.
- Added scenario-to-prompt consistency checks.
- Strengthened attack-neutrality and benchmark-language controls.

## Generation code
- Test path now uses automatic retry logic.
- Default temperature lowered to 0.4.
- Default top_p set to 0.9.
- Default retries increased to 5.
- Resume-safe output remains enabled.

## Pilot design
- V1 pilot: first 10 rows, all HC1.
- V2 pilot: 18 stratified rows, 2 from every HC category.
- V2 target: 90 candidate tasks.

## Validation
- Added optional semantic validator.
- Validator scores harmful intent, chemistry dependency, HC fit, HD fit, OT fit,
  chemistry plausibility, scenario consistency, and jailbreak readiness.
- Validator returns ACCEPT, REVISE, or REJECT.

## GitHub
- Added GitHub-first Colab setup.
- Added optional GitHub checkpoint helper.
