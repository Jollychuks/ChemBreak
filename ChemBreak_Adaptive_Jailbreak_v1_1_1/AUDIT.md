# Offline build audit

- Fresh revision namespace: CB-ADAPTIVE-JAILBREAK-V1.1.1
- Repository target: https://github.com/Jollychuks/ChemBreak
- Repository subdirectory: ChemBreak_Adaptive_Jailbreak_v1_1_1
- GitHub to Colab Enterprise notebook implemented
- No prior task-bank outputs modified
- Production task bank remains in restricted local/GCS storage
- Controlled C0, C1, C2 retained
- Adaptive C3 retained
- Four chemistry-specific route families retained
- Initial attack assets frozen before target execution
- Target response loops back to attacker only in C3
- Judge feedback is not given to attacker
- Target-query budgets counted explicitly
- Attacker role separated from safety judge
- GPT-OSS 120B configured as safety/goal judge
- Gemini 2.5 Pro configured as chemistry judge
- Deterministic Chemistry Domain Verifier added
- RDKit tagged-SMILES validation added
- Verifier does not treat UNVERIFIABLE as failure
- Verifier contradictions escalate to adjudication
- Llama 4 Maverick configured as adjudicator in us-east5
- Open-model JSON mode added through Vertex OpenAI-compatible API
- Preflight checks all required model roles before attack execution
- Preflight gives a Llama EULA/region hint on failure
- Raw transcripts separated from public metrics
- Alignment breach and effective chemical breach recorded separately
- Public metrics include adjudication, disagreement, and verifier contradiction rates
- GCS checkpoint sync retained
- Stable run IDs and resumability retained
- Visible progress and ETA retained
- Target models loaded one at a time and reused within target blocks
- No live target-model or jailbreak calls made during packaging
- Python compilation passed after model/verifier revision
- Offline unit tests passed: 9/9

## v1.1.1 preparation repair

- Added structural validation retries for graph, C1 repeated-single, and C2 fixed-multi frozen assets.
- Maximum is 3 total model generations per asset by default, configurable under `asset_preparation.max_generation_attempts`.
- Structural defects are fed back explicitly before regeneration.
- Bare-list JSON is normalized only when its intended wrapper is unambiguous.
- Fixed C2 schema example now reflects the configured query count rather than a hard-coded three-query example.
- Completed v1.1 assets can be imported read-only into the v1.1.1 namespace.
- Prepare now writes `prepare_summary.json` and exits nonzero when selected tasks still lack complete assets.
- Existing completed assets are skipped on rerun.
- Python compilation passed after the patch.
- Offline unit tests passed: 9/9.
