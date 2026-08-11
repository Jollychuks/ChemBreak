"""
Optional GitHub checkpoint helper for ChemBreak Colab Generator V2.

This is NOT an LLM API and has no LLM usage charge.
It only authenticates to GitHub so Colab can push generated CSV checkpoints.

Pass a GitHub personal access token at runtime. Do not save the token in the repo.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from urllib.parse import quote


def _run(args, cwd=None):
    return subprocess.run(
        args, cwd=cwd, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def checkpoint_to_github(
    repo_dir,
    files,
    commit_message,
    token,
    branch="main",
    git_name="ChemBreak Colab",
    git_email="chembreak-colab@local",
):
    repo_dir = Path(repo_dir)

    if not token:
        raise ValueError("A GitHub token is required for pushing.")

    origin = _run(["git", "remote", "get-url", "origin"], cwd=repo_dir).stdout.strip()

    # Accept HTTPS or SSH GitHub origin and extract owner/repo.
    m = re.search(r"github\.com[/:]([^/]+)/([^/]+?)(?:\.git)?$", origin)
    if not m:
        raise ValueError(f"Could not parse GitHub origin: {origin}")

    owner, repo = m.group(1), m.group(2)
    auth_url = f"https://x-access-token:{quote(token, safe='')}@github.com/{owner}/{repo}.git"

    _run(["git", "config", "user.name", git_name], cwd=repo_dir)
    _run(["git", "config", "user.email", git_email], cwd=repo_dir)

    relative_files = [str(Path(f).resolve().relative_to(repo_dir.resolve())) for f in files]
    _run(["git", "add", "--"] + relative_files, cwd=repo_dir)

    status = _run(["git", "status", "--porcelain"], cwd=repo_dir).stdout.strip()
    if not status:
        print("No checkpoint changes to commit.")
        return

    _run(["git", "commit", "-m", commit_message], cwd=repo_dir)
    _run(["git", "push", auth_url, f"HEAD:{branch}"], cwd=repo_dir)
    print("Checkpoint pushed to GitHub.")
