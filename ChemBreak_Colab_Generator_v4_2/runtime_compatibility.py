from __future__ import annotations


EXPECTED = {
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "torchaudio": "2.11.0",
    "transformers": "5.15.0",
}


def check_runtime():
    """
    Post-bootstrap in-kernel verification.

    V4.2 runs the repair/bootstrap in child processes first. Only after that
    succeeds does this function import the runtime packages into the notebook.
    """
    import torch
    import torchvision
    import torchaudio
    import torch.distributed._functional_collectives
    import transformers
    import accelerate
    import bitsandbytes

    versions = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torchaudio": torchaudio.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }

    for name, expected in EXPECTED.items():
        if not str(versions[name]).startswith(expected):
            raise RuntimeError(
                f"{name} version mismatch: expected {expected}, "
                f"found {versions[name]}"
            )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch is internally consistent but CUDA is unavailable. "
            "Use a Colab GPU runtime."
        )

    print("ChemBreak V4.2 in-kernel runtime verification: PASS")
    for key, value in versions.items():
        print(f"{key}: {value}")

    return versions
