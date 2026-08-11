# ChemBreak V4 Colab Resource and Session Guide

## Why V4 should usually be run one family at a time

V4 compares four model families, but the models are not intended to remain
loaded together.

A Colab T4-class runtime has limited GPU memory and local disk. Downloading
and caching all four model repositories in one long session can consume much
more disk than a single-family run.

The safest workflow for the full pilot is therefore:

1. Run generator family A.
2. Confirm its candidates were pushed to GitHub.
3. End the runtime if desired.
4. Start a fresh runtime and run generator family B.
5. Repeat for C and D.
6. Then run judge family A in a fresh session.
7. Repeat for judge families B, C and D.

The V4 output CSVs are append-only and resume-safe.

## Selecting one family in the notebook

For generation, replace:

`GENERATOR_FAMILIES_TO_RUN = list(config["generation_families"])`

with, for example:

`GENERATOR_FAMILIES_TO_RUN = ["A"]`

For judging, replace:

`JUDGE_FAMILIES_TO_RUN = list(config["judge_families"])`

with, for example:

`JUDGE_FAMILIES_TO_RUN = ["A"]`

The same process works for B, C or D.

## GitHub checkpointing for split sessions

If you plan to delete the Colab runtime after a family finishes, enable
GitHub checkpointing in the active config:

`"enabled": true`

The notebook will ask for your GitHub personal access token at runtime.

The token is not written to disk.

V4 performs periodic checkpoints and also forces one final push at the end of
every successfully completed generator-family or judge-family run.

Before deleting a runtime, verify that the relevant output CSV and progress
files are visible in GitHub.

## Hugging Face authentication

Some model repositories may require that you accept repository terms and
authenticate.

If needed, set:

`USE_HF_TOKEN = True`

in the notebook and enter the Hugging Face token only when prompted.

Do not put Hugging Face or GitHub tokens in:
- `run_config.json`
- `model_registry.json`
- the notebook source
- GitHub commits

## Smoke test first

Run `run_config_smoke.json` before the full comparison.

The smoke run has its own `outputs_smoke/` directory and therefore cannot
contaminate the full `outputs/` directory.

## Do not manually merge CSVs

When using GitHub checkpoints and resume, let the V4 code append and resume
using its IDs.

Do not copy rows between candidate or judgment CSVs by hand unless you are
performing a deliberate recovery and have checked the experiment IDs.


## V4.2 PyTorch rule

Do not run a separate `pip install torch`, `pip install torchvision`, or
`pip install torchaudio` before V4.2.

`bootstrap_colab.py` is the only component that manages the PyTorch trio.

If Cell 1 says model-runtime modules are already loaded, restart the Colab
session before continuing. This prevents replacing package files while old
modules remain in kernel memory.
