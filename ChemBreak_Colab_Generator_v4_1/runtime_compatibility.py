from __future__ import annotations


def check_runtime():
    """
    Verify that the active PyTorch installation is internally consistent.

    Run this only AFTER the dependency installation cell.
    """
    import torch
    import torch._utils

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    if not hasattr(torch._utils, "_chunk_or_narrow_cat"):
        raise RuntimeError(
            "The active Colab kernel is using an inconsistent PyTorch "
            "installation. This commonly happens when PyTorch-related "
            "packages are changed after torch was already imported. "
            "Use Runtime > Restart session, then rerun the notebook "
            "from the beginning. Do not delete the runtime."
        )

    try:
        import torchvision
        print("torchvision:", torchvision.__version__)
    except Exception as exc:
        raise RuntimeError(
            "torchvision could not be imported with the active PyTorch "
            f"installation: {exc}. Restart the Colab session first. "
            "If the problem remains in a fresh session, reinstall a "
            "matched torch/torchvision pair before continuing."
        ) from exc

    print("PyTorch runtime consistency check: PASS")
    return {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
