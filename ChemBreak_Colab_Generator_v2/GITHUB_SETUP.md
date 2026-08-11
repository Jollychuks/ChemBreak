# GitHub Setup

V2 can use GitHub as the source for the project files instead of Google Drive.

## Public repository

The notebook can clone a public repository without any token.

Example:

```python
REPO_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO.git"
PROJECT_SUBDIR = "ChemBreak_Colab_Generator_v2"
```

If the V2 files are at the repository root:

```python
PROJECT_SUBDIR = "."
```

## Private repository

Cloning or pushing a private repository requires GitHub authentication.

A GitHub token is not an LLM API key and does not create per-token LLM charges.

Do not hard-code the token inside the repository.

## Generated outputs

Colab runtime storage disappears when the runtime is destroyed.

You have three options:

1. Periodically push `candidate_tasks.csv` and progress files to GitHub.
2. Download output files to your computer.
3. Use Google Drive only for output persistence.

V2 includes `github_checkpoint.py` for optional GitHub pushes.

In the notebook, use `getpass()` to enter the token at runtime so it is not written into the notebook.

Recommended checkpoint files:

- `candidate_tasks.csv`
- `generation_progress.csv`
- `generation_errors.jsonl`
- `candidate_tasks_validated.csv`
- `validation_progress.csv`
- `validation_errors.jsonl`
