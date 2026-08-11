# ChemBreak V4.2 Changelog

## V4.2

- Replaced the V4.1 PyTorch marker check with a full child-process bootstrap.
- Added `bootstrap_colab.py`.
- Added a clean-kernel guard before any PyTorch/model-runtime import.
- Added a child-process probe of torch, torchvision, torchaudio and
  `torch.distributed._functional_collectives`.
- Added automatic repair to the official matched CUDA 12.8 stack when the
  probe fails or the versions differ:
  - torch 2.11.0
  - torchvision 0.26.0
  - torchaudio 2.11.0
- Added complete child-process verification of Transformers, Accelerate,
  bitsandbytes, pandas and Hugging Face Hub.
- Changed `runtime_compatibility.py` into a post-bootstrap in-kernel verifier.
- Kept `requirements_colab.txt` free of PyTorch packages.
- Corrected all package-folder references to
  `ChemBreak_Colab_Generator_v4_2`.
- Corrected the notebook filename to
  `ChemBreak_Colab_Generator_V4_2.ipynb`.
- Kept the V4 generator, judge, matrix, taxonomy, scenario, scoring,
  aggregation, human-calibration and checkpoint methodology unchanged.
