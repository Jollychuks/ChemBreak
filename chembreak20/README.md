# ChemBreak20

ChemBreak20 is a controlled hierarchical-MDP safety-evaluation harness for a fixed 24-task chemistry-language-model panel. It separates **strategy learning** from **language realization**:

- deterministic Python chooses and updates abstract actions;
- the Attack LLM realizes the selected action as a candidate message;
- an immutable task lock prevents goal drift;
- the Judge LLM supplies structured measurements;
- Python alone applies the final success gate;
- abstract action routes, not exact prompts, persist across learning epochs.

## Target dialogue formats

CB20 uses each target's documented/native dialogue format rather than applying one generic prompt wrapper to both models. ChemDFM uses its `[Round n] / Human / Assistant` format. ChemLLM uses the tokenizer-native InternLM2 chat template, with a model-card-compatible fallback if the tokenizer does not expose `apply_chat_template`.

## Phases per target

1. **Baseline:** original prompt, one target query, fresh context.
2. **Learning Epoch 1:** up to 4 turns/task, epsilon 0.30.
3. **Learning Epoch 2:** fresh target context, retained controller memory, epsilon 0.20.
4. **Learning Epoch 3:** fresh target context, retained controller memory, epsilon 0.15.
5. **Freeze:** Q-values and route memory become immutable.
6. **Terminal exploitation:** fresh target context, up to 4 target turns. The first two attempts are informed by the two strongest distinct learned routes; remaining attempts are frozen-policy syntheses. No learning occurs.

ChemDFM and ChemLLM have independent Q-tables, route memories, checkpoints, and output directories.

See `docs/METHODOLOGY.md` for the formal protocol and `docs/OUTPUTS.md` for result files.
