# ChemBreak37 | GitHub-to-Colab release

**Version:** 37.0.0  
**Source repository:** https://github.com/Jollychuks/ChemBreak

## Judging behavior

GPT-5.6 Sol is the primary cumulative CHCS judge. The local Qwen3.6-35B-A3B (Q4_K_M, thinking enabled) is used **only when GPT cannot return a valid judgment**. A valid primary judgment does not trigger Qwen. Both judges failing leaves an unresolved judgment, not CHCS 1. No extra latest-turn CHCS is introduced. The existing five-level cumulative rubric, state fields, and reward calculations are retained.

Qwen uses a 3,072-token reasoning budget, bounded retries, and score/class/reason consistency checks. These checks improve mechanical validation; they do not independently establish scientific accuracy or guarantee fewer false-positive CHCS 5 scores.

## Contents

- `configs/config.cb37.yaml`: default experiment settings
- `data/prompts.csv`: original 28 benchmark behaviors, unchanged
- `data/CB37_prompts28_manifest_v1.csv`: ChemBreak37 task selection and checksums
- `data/CB37_prompts28_lock_v1.json`: ChemBreak37 immutable selection lock
- `notebooks/chembreak37_Cloud_Notebook.ipynb`: GitHub-to-Colab notebook
- `src/chembreak37/`: required Python package modules
- `pyproject.toml`, `requirements-cloud-ml.txt`: dependency declarations

No old run outputs, prior-version folders, unused standalone comparison artifacts, audit reports, or development tests are included in this runtime release.

## GitHub → Colab setup

1. **Extract** the release ZIP locally. Upload the entire extracted `chembreak37` directory into the root of the ChemBreak GitHub repository and commit it to the `main` branch.
2. Open `chembreak37/notebooks/chembreak37_Cloud_Notebook.ipynb` in Google Colab. Use a fresh runtime.
3. Cell 2 is preconfigured with `REPOSITORY_URL = "https://github.com/Jollychuks/ChemBreak.git"`, `REPOSITORY_BRANCH = "main"`, and `PROJECT_SUBDIR = "chembreak37"`. Change these only if you use another branch or repository.
4. Cell 3 **always fetches the committed GitHub version**. It does not load an old local ZIP or stale pre-extracted folder.
5. Start with `DRY_RUN = True` to verify the local workflow without live model queries. For live operation, use an appropriate GPU runtime and provide required credentials through secure prompts/environment variables. Never commit API keys.
6. Review preflight results. Keep private transcript exports private, and start a new `RUN_ID` when changing release configuration or run mode.

## Reproducibility note

The source behavior text in `data/prompts.csv` is unchanged, with SHA-256 `80c10ea78cf859174e5eb83d24c1bfe49b73b5adeab8b7b59c3045cb201c8434`.

All task IDs use `CB37P-0001` through `CB37P-0028`. Because the existing task-seed calculation incorporates task IDs, changing those identifiers **changes task-derived random seeds versus archived experiments**, even if the top-level seed is unchanged. This release does not claim to be seed-identical with previous releases. Judge-only comparisons should use identical archived responses and should account for this distinction.

The original model/runtime evaluation path has been checked using offline unit tests and mock runs. A live Colab run, actual GPT API behavior, cloud authentication, local GGUF server readiness, and scientific correctness of assigned CHCS scores have not been validated here.
