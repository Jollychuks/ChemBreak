from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


TORCH_VERSION = "2.11.0"
TORCHVISION_VERSION = "0.26.0"
TORCHAUDIO_VERSION = "2.11.0"
PYTORCH_INDEX = "https://download.pytorch.org/whl/cu128"


def _run(
    args: List[str],
    *,
    check: bool = False,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    kwargs = {
        "text": True,
        "check": check,
    }

    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.STDOUT

    return subprocess.run(args, **kwargs)


def kernel_is_clean() -> Tuple[bool, List[str]]:
    """
    This check does not import PyTorch.

    If a previous notebook already imported torch-related modules into the
    active kernel, package replacement would leave stale modules in memory.
    In that case the user should restart the Colab session before setup.
    """
    prefixes = (
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "accelerate",
        "bitsandbytes",
    )

    loaded = sorted(
        name
        for name in sys.modules
        if name == prefixes
        or name.startswith(tuple(p + "." for p in prefixes))
    )

    return (len(loaded) == 0, loaded)


def assert_clean_kernel() -> None:
    clean, loaded = kernel_is_clean()

    if clean:
        print("Kernel package-import guard: PASS")
        return

    preview = ", ".join(loaded[:12])
    if len(loaded) > 12:
        preview += ", ..."

    raise RuntimeError(
        "This Colab session already has PyTorch/model-runtime modules in "
        f"memory: {preview}\n\n"
        "Choose Runtime > Restart session, then run V4.2 again from Cell 1. "
        "Do not continue installing packages in the current session."
    )


def _stack_probe_code() -> str:
    return r"""
import json

result = {
    "ok": False,
    "torch": None,
    "torchvision": None,
    "torchaudio": None,
    "cuda": None,
    "distributed_collectives": False,
    "error": None,
}

try:
    import torch
    result["torch"] = torch.__version__
    result["cuda"] = torch.version.cuda

    import torchvision
    result["torchvision"] = torchvision.__version__

    import torchaudio
    result["torchaudio"] = torchaudio.__version__

    # This is the actual import path that failed in the earlier runtime.
    import torch.distributed._functional_collectives
    result["distributed_collectives"] = True

    result["ok"] = True
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"

print(json.dumps(result))
"""


def probe_torch_stack() -> Dict[str, object]:
    """
    Probe PyTorch in a CHILD PROCESS so the notebook kernel does not import it.
    """
    proc = _run(
        [
            sys.executable,
            "-c",
            _stack_probe_code(),
        ],
        capture=True,
    )

    output = (proc.stdout or "").strip().splitlines()

    if not output:
        return {
            "ok": False,
            "error": "PyTorch probe returned no output.",
        }

    try:
        return json.loads(output[-1])
    except Exception:
        return {
            "ok": False,
            "error": "Could not parse PyTorch probe output.",
            "raw_output": "\n".join(output[-20:]),
        }


def expected_stack(result: Dict[str, object]) -> bool:
    return (
        bool(result.get("ok"))
        and str(result.get("torch", "")).startswith(TORCH_VERSION)
        and str(result.get("torchvision", "")).startswith(TORCHVISION_VERSION)
        and str(result.get("torchaudio", "")).startswith(TORCHAUDIO_VERSION)
        and bool(result.get("distributed_collectives"))
    )


def repair_torch_stack() -> None:
    """
    Install the official matched CUDA 12.8 PyTorch trio.

    This is run only while the notebook kernel has not imported torch.
    """
    print("\nRepairing the PyTorch stack with the official matched CUDA 12.8 wheels.")
    print(
        f"torch={TORCH_VERSION}, "
        f"torchvision={TORCHVISION_VERSION}, "
        f"torchaudio={TORCHAUDIO_VERSION}"
    )

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-cache-dir",
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
        f"torchaudio=={TORCHAUDIO_VERSION}",
        "--index-url",
        PYTORCH_INDEX,
    ]

    proc = _run(cmd, capture=False)

    if proc.returncode != 0:
        raise RuntimeError(
            "The official PyTorch repair command failed. "
            "Review the pip output above before continuing."
        )


def ensure_torch_stack() -> Dict[str, object]:
    """
    Keep an already-correct exact stack, otherwise repair it once and verify.
    """
    assert_clean_kernel()

    before = probe_torch_stack()

    print("\nPyTorch child-process probe BEFORE repair:")
    print(json.dumps(before, indent=2))

    if expected_stack(before):
        print("\nMatched PyTorch CUDA 12.8 stack already present. Repair not needed.")
        return before

    repair_torch_stack()

    after = probe_torch_stack()

    print("\nPyTorch child-process probe AFTER repair:")
    print(json.dumps(after, indent=2))

    if not expected_stack(after):
        raise RuntimeError(
            "PyTorch repair completed but the child-process verification "
            "still failed. Do not continue to model loading. "
            "Choose Runtime > Disconnect and delete runtime, reconnect to a "
            "fresh GPU runtime, and run V4.2 from Cell 1."
        )

    print("\nPyTorch stack repair and verification: PASS")
    return after


def install_chembreak_dependencies(
    project_dir: str | Path,
) -> None:
    """
    Install non-PyTorch ChemBreak dependencies only after the torch stack is
    verified in a child process. The notebook kernel is still torch-free here.
    """
    project_dir = Path(project_dir)
    requirements = project_dir / "requirements_colab.txt"

    if not requirements.exists():
        raise FileNotFoundError(requirements)

    print("\nInstalling ChemBreak non-PyTorch dependencies.")

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "-r",
        str(requirements),
    ]

    proc = _run(cmd, capture=False)

    if proc.returncode != 0:
        raise RuntimeError(
            "ChemBreak dependency installation failed. "
            "Review the pip output above."
        )


def full_child_process_probe() -> Dict[str, object]:
    """
    Verify the complete runtime before the notebook imports V4 modules.
    """
    code = r"""
import json

result = {"ok": False, "versions": {}, "error": None}

try:
    import torch
    import torchvision
    import torchaudio
    import torch.distributed._functional_collectives
    import transformers
    import accelerate
    import bitsandbytes
    import pandas
    import huggingface_hub

    result["versions"] = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torchaudio": torchaudio.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "pandas": pandas.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    result["ok"] = True
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"

print(json.dumps(result))
"""

    proc = _run(
        [
            sys.executable,
            "-c",
            code,
        ],
        capture=True,
    )

    lines = (proc.stdout or "").strip().splitlines()

    if not lines:
        return {
            "ok": False,
            "error": "Full runtime probe returned no output.",
        }

    try:
        return json.loads(lines[-1])
    except Exception:
        return {
            "ok": False,
            "error": "Could not parse full runtime probe output.",
            "raw_output": "\n".join(lines[-30:]),
        }


def bootstrap(project_dir: str | Path) -> Dict[str, object]:
    """
    Complete pre-import setup for a Colab session.
    """
    assert_clean_kernel()

    torch_info = ensure_torch_stack()

    # Still no torch import in the notebook kernel.
    install_chembreak_dependencies(project_dir)

    result = full_child_process_probe()

    print("\nComplete child-process runtime verification:")
    print(json.dumps(result, indent=2))

    if not result.get("ok"):
        raise RuntimeError(
            "The complete V4.2 runtime verification failed. "
            "Do not continue to model loading. Review the error above."
        )

    print("\nChemBreak V4.2 environment bootstrap: PASS")
    return result
