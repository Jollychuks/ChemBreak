# ChemBreak V4.1 Changelog

## V4.1

- Fixed the Colab mixed-PyTorch import failure involving `_chunk_or_narrow_cat`.
- GPU detection now uses `nvidia-smi` before dependency installation instead of importing torch.
- Removed `torchvision` from `requirements_colab.txt`.
- V4.1 does not upgrade torch, torchvision or torchaudio.
- Removed `-U` from the dependency installation command.
- Added `runtime_compatibility.py`.
- Added a PyTorch/torchvision consistency check before the ChemBreak modules are imported.
- Added a clear restart-session instruction if a mixed runtime is detected.
- Preserved the V4 four-family generation, judging, scenario, checkpoint and comparison methodology unchanged.

- Added four distinct open-weight model families.
- Added full 4×4 generator-judge comparison.
- Added central `model_registry.json`.
- Added one persisted scenario plan shared by all generator families.
- Removed scenario selection and scenario metadata output from generator LLMs.
- Added family-prefixed candidate IDs.
- Standardized corresponding numeric generation seeds across families.
- Added blind judging that hides generator identity and generator-authored metadata.
- Removed model-generated ACCEPT/REVISE/REJECT decisions.
- Added Python score-to-decision rule.
- Added deterministic shuffled judge order.
- Added rolling uniform-score judge health check.
- Added immediate candidate and judgment CSV checkpointing.
- Added optional periodic GitHub pushes.
- Added all-judge and cross-family generator rankings.
- Added generator-by-judge cross matrix.
- Added self-family judge-bias analysis.
- Added pairwise exact judge agreement and Cohen's kappa.
- Added blinded human-review sample.
- Added human-calibrated judge ranking.
- Added direct human-calibrated generator ranking.
- Added experiment manifest with hashes, Git commit, package versions and GPU metadata.
- Added environment freeze.
- Added separate smoke and full-pilot output directories.
