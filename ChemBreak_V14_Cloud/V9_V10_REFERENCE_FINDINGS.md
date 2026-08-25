# V9 and V10 findings used to design V14

The V14 architecture was informed by the attached V9 and V10 test checkpoints.

## Observed checkpoint metrics

| Metric | V9 | V10 |
|---|---:|---:|
| Final task-bank completion | 5/9 | 8/9 |
| Initial candidates recorded | 23 | 27 |
| Initial deterministic validation passes | 13/23 | 18/27 |
| Gemini 3.1 Pro generation passes | 4/5 | 8/9 |
| Llama 4 generation passes | 6/9 | 5/9 |
| gpt-oss-120B generation passes | 3/9 | 5/9 |
| Repair passes | 3/10 | 4/9 |
| Refill passes | 0/3 | 0/2 |
| Final tasks sourced from Gemini 3.1 Pro | 1/5 | 5/8 |

## Judge reliability signal

The V9 error log contained 8 Gemini 2.5 Pro judge errors. The V10 error log contained 11 Gemini 2.5 Pro judge errors, primarily around structured judge output. This motivated V14's explicit structured-output retry path and the rule that a parser/API failure is not a vote.

## Design consequences

1. Gemini 3.1 Pro becomes the sole bulk generator because it showed the strongest raw deterministic-validation rate in V10.
2. Diversity is produced through three controlled candidate language profiles instead of three generator families.
3. Repair moves to Gemini 3.1 Pro.
4. Refill receives exact failure history and produces two fresh candidates per cycle.
5. Both judges evaluate every active candidate set concurrently so a single judge cannot silently favor a safer or weaker task.
6. Judge prompts explicitly separate benchmark-quality evaluation from the question of whether a deployed assistant should comply.
7. A single surviving candidate is not discarded. V14 attempts targeted pre-judge refill and then uses dual qualification if necessary.

These checkpoint observations are development evidence, not claims that one model is universally better than another.
