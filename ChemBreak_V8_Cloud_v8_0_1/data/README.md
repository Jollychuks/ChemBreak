# V8 data directory

The repository intentionally does not depend on any V1 to V7 ChemBreak generated task files.

On the first `bootstrap` run, the pipeline creates:
- `entities_v8.csv`
- `external_reference_behaviors.csv`

The source URLs and their purposes are recorded in `source_manifest.json`.

Generated run data belongs under `outputs/<run_name>/` and should normally be kept out of Git except for small review samples.
