# ChemBreak 12 Runbook

## Step A — Create a fresh CB12 environment

Use a fresh project directory and do not point CB12 to ChemBreak 7-11 checkpoint or output folders. The CB12 partition code does not read any previous ChemBreak checkpoint.

Install:

```bash
python -m pip install -e .
```

## Step B — Verify the source bank

```bash
chembreak12 --root . verify
```

Do not proceed if verification fails.

## Step C — Confirm the fixed split sizes

Expected:

```text
Train   241
Test1    50
Test2    50
Test3    50
Test4    50
Reserve  59
```

## Step D — Training access

Before a training run:

```bash
chembreak12 guard-run --mode train --split Train
```

Then load only `data/splits/CB12_Train.csv`.

Do not open Test1-Test4 in training code simply to inspect performance while tuning. Repeated test inspection turns the test set into development data.

## Step E — Freeze model/policy choices

After training/development is complete, save the policy/configuration and its checksum before evaluating Test1-Test4. Changes made after looking at held-out results should be recorded as a new experiment revision.

## Step F — Held-out evaluation

Use one held-out partition at a time:

```bash
chembreak12 guard-run --mode eval --split Test1
```

The guard rejects Train in ordinary evaluation mode and rejects Test partitions in training mode.

## Step G — Reserve use

Reserve is not a fifth ordinary test set. It is a contingency pool for cases defined in the methodology (for example, a source task later found to be unusable for a documented non-performance reason).

Access requires a reason:

```bash
chembreak12 guard-run --mode reserve --split Reserve --reason "documented reason"
```

A Reserve task should never be substituted merely because a test task was difficult or unsuccessful.

## Step H — Archive experiment metadata

For each experiment revision archive at minimum:

- CB12 partition lock;
- partition manifest;
- source-bank checksum;
- model identifiers/revisions;
- training configuration;
- evaluation configuration;
- random seeds;
- result files;
- code commit hash.
