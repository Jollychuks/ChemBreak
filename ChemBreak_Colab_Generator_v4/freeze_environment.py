from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path


def freeze_environment(output_path):
    output_path = Path(output_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "freeze",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    text = (
        f"Python: "
        f"{platform.python_version()}\n\n"
        "pip freeze:\n"
        f"{result.stdout}"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        text,
        encoding="utf-8",
    )

    return output_path
