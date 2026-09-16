# ChemBreak 17 — v17.0.0

CB17 keeps the same fixed 24-task panel and Baseline/E1/E2/E3/Optimized phase structure while adding an explicit evidence-memory layer.

Key changes:

- GPT-5.6 Sol is the **attack LLM** through the OpenAI Responses API;
- Gemini 2.5 Flash remains the single **judge LLM**;
- persistent evidence memory stores an exact-text `realization_id` and a separate action-conditioned `candidate_id`;
- repeated success strengthens evidence; later failure weakens but does not erase prior success;
- formal ranking uses Wilson reliability, observation support, mean reward quality, and mean Q evidence;
- learning exploitation can replay a previously successful exact realization, while epsilon exploration still generates fresh realizations;
- freeze snapshot makes Q-values, evidence memory, and candidate rankings immutable before Optimized;
- Optimized uses exact frozen replay first, then frozen-Q/fresh-attack-LLM fallback;
- attempted candidate masking prevents exact-candidate cycling;
- failed fresh fallback actions are avoided when an untried alternative exists;
- outputs add ASR@1, ASR@4, success retention, recovery rate, evidence coverage, `evidence_memory.csv`, and `candidate_rankings.csv`;
- no matcher, train/test split, second judge, or adjudicator is included;
- exactly one cloud notebook is included under `notebooks/`.
