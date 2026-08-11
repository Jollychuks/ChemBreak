# V3 to V4 Changes

| Area | V3 | V4 |
|---|---|---|
| Generator models | One Qwen model | Four independent model families |
| Judge models | Same local model | Every candidate is judged by all four families |
| Design | Single generator/judge path | Full 4×4 crossed comparison |
| Scenario assignment | Python-controlled | Python-controlled and persisted once for all families |
| Scenario fairness | Stable per row | Identical corresponding HC+HD+OT+SC conditions across A-D |
| Generator output | Model returned scenario field and Python overwrote it | Model does not return scenario metadata at all |
| Candidate IDs | `GM001-C0001` | `A-GM001-C0001`, `B-GM001-C0001`, etc. |
| Generator seed | Local model seed | Same corresponding numeric seed across A-D |
| Judge visibility | Validator received candidate metadata | Judge is blind to generator identity and generator-authored metadata |
| Judge output | Scores and model decision | Eight scores plus reason only |
| Final judge decision | Model-selected | Python-derived with one fixed rule |
| Judge failure monitoring | None | Rolling uniform-score health check |
| Judge order | Dataset order | Deterministically shuffled per judge family |
| Generator ranking | Not applicable | Separate all-judge and cross-family rankings |
| Self-family bias | Not measured | Explicitly measured |
| Judge agreement | Not measured | Exact agreement, Cohen's kappa, score distance |
| Human calibration | Ad hoc | Blinded stratified human calibration |
| Human generator comparison | None | Direct human-calibrated generator ranking |
| Reproducibility | Prompt/model metadata | Experiment manifest, hashes, Git commit, versions and environment freeze |
| GitHub checkpoint | Manual/optional | Optional periodic generation and judgment checkpoint |
| Runtime | One model | Four models loaded sequentially to fit constrained GPUs |
| Smoke test | None | Separate 8-candidate, 32-judgment compatibility run |
