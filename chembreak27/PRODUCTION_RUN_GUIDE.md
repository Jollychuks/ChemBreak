# ChemBreak27 Production Run Guide

## 1. Validate package

```bash
python scripts/validate_package.py
pytest -q
```

## 2. Use the Colab notebook

Open `notebooks/chembreak27_Cloud_Notebook.ipynb` and run from the first cell downward.

The notebook installs the package, creates an isolated storage root, performs dataset/config/provider preflight, and runs ChemDFM and ChemLLM separately.

## 3. Resume behavior

Re-running the same experiment revision resumes completed tasks and episodes. Incomplete episodes are rolled back to their pre-episode learning snapshot. Their issued target calls remain in the audit database with no committed-turn link and are excluded from all scientific success calculations.

If methodological settings change, use a new `experiment_revision` and a new run directory.

## 4. Expected task artifacts

Each completed task has a policy directory containing epoch snapshots plus frozen policy and routes. The target run directory holds the SQLite audit store, public release exports, and private internal exports.

## 5. Isolation check

Every task directory contains `task_isolation_manifest.json` with:

```text
loaded_prior_task_policy: false
loaded_prior_task_routes: false
policy_scope: current_task_only
route_scope: current_task_only
```

## 6. Public versus private output

Public release:
- prompt/response text redacted
- route actions/statistics only
- aggregate metrics

Private audit:
- raw benchmark prompts
- candidate prompts
- target responses
- judge provenance
- successful trajectory text
- SQLite state database if separately archived

Keep private audit files restricted.

## 7. Live mode

The committed configuration intentionally defaults to `dry_run: true`. In the notebook, set `LIVE = True`, provide the authorized Google Cloud project and OpenAI API key, select a GPU runtime, and run every cell from the top. The notebook writes a separate runtime configuration and does not modify the locked package configuration.
