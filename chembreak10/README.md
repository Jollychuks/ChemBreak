# ChemBreak10 — Adaptive MDP Jailbreak Study

Multi-turn adaptive jailbreak evaluation for chemistry language models
using a Markov Decision Process Q-policy.

## Research targets
- **ChemDFM** — OpenDFM/ChemDFM-v1.5-8B (LLaMA-8B base)
- **ChemLLM** — AI4Chem/ChemLLM-7B-Chat-1_5-SFT (InternLM-2 base)

## Condition
C3_ADAPTIVE_MDP only. C0/C1/C2 baselines excluded. Add them later for comparison.

## How to run
1. Upload `chembreak10/` to your GitHub repo and push.
2. Open `notebooks/chembreak10_Cloud_Notebook.ipynb` in Colab Enterprise.
3. Fill in `PROJECT_ID` in cell 1.
4. Run all cells top to bottom.

## Phase order

| Phase | Tasks | Policy | Verification | Purpose |
|-------|-------|--------|-------------|---------|
| development | 48 | train | screened | Train Q-policy |
| pilot | 100 | frozen | strict | First real evaluation |
| full_bank | 200 | frozen | strict | Final reportable result |

Change `PHASE` in cell 1 to switch phases.

## Speed settings (development)
- verification_mode: screened (Observer every turn; full pipeline on candidates only)
- max_new_tokens: 256 (refusals are short)
- target_query_budget: 6 turns
- chemistry_verifier: gemini-2.5-flash

## Output
Results in `/content/chembreak10_storage/runs/CB10_<phase>_<signature>/release/`
