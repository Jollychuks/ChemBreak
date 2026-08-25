# V14 runtime data

The GitHub package intentionally does not contain a generated task bank.

At runtime, the bootstrap stage creates fresh source snapshots in the selected V14 output directory:

- `entities_v14.csv` from IDEA-XL ChemSafety `substances.json`, plus taxonomy system targets for domains that require system-level entities.
- `external_reference_behaviors_v14.csv` from the HarmBench `chemical_biological` subset for similarity checking only.
- `source_snapshot_manifest.json` with source URLs and SHA-256 hashes.

Prior ChemBreak generated tasks are not used as V14 generation input.
