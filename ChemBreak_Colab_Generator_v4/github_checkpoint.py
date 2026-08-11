"""
GitHub checkpoint helper for ChemBreak V4.

The GitHub personal access token is entered at runtime.
This module never writes the token to disk.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import quote


def _run(args, cwd=None):
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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
        raise ValueError(
            "A GitHub token is required "
            "for pushing."
        )

    origin = _run(
        [
            "git",
            "remote",
            "get-url",
            "origin",
        ],
        cwd=repo_dir,
    ).stdout.strip()

    match = re.search(
        r"github\.com[/:]"
        r"([^/]+)/"
        r"([^/]+?)(?:\.git)?$",
        origin,
    )

    if not match:
        raise ValueError(
            "Could not parse GitHub origin: "
            f"{origin}"
        )

    owner = match.group(1)
    repo = match.group(2)

    auth_url = (
        "https://x-access-token:"
        f"{quote(token, safe='')}"
        f"@github.com/{owner}/{repo}.git"
    )

    _run(
        [
            "git",
            "config",
            "user.name",
            git_name,
        ],
        cwd=repo_dir,
    )

    _run(
        [
            "git",
            "config",
            "user.email",
            git_email,
        ],
        cwd=repo_dir,
    )

    relative_files = []

    for file in files:
        path = Path(file)

        if path.exists():
            relative_files.append(
                str(
                    path.resolve()
                    .relative_to(
                        repo_dir.resolve()
                    )
                )
            )

    if not relative_files:
        print(
            "No checkpoint files "
            "exist yet."
        )
        return

    _run(
        [
            "git",
            "add",
            "--",
            *relative_files,
        ],
        cwd=repo_dir,
    )

    status = _run(
        [
            "git",
            "status",
            "--porcelain",
        ],
        cwd=repo_dir,
    ).stdout.strip()

    if not status:
        print(
            "No checkpoint changes "
            "to commit."
        )
        return

    _run(
        [
            "git",
            "commit",
            "-m",
            commit_message,
        ],
        cwd=repo_dir,
    )

    _run(
        [
            "git",
            "push",
            auth_url,
            f"HEAD:{branch}",
        ],
        cwd=repo_dir,
    )

    print(
        "Checkpoint pushed "
        "to GitHub."
    )
