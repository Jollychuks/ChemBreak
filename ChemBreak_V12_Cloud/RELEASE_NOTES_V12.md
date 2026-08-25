# ChemBreak V12 Release Notes

V12 is a reliability and runtime-efficiency release built from V11 without changing the locked taxonomy or the two-independent-judge methodology.

## Reliability fix

V11 showed 9 successful gpt-oss Judge A decisions and 36 Gemini 2.5 Pro Judge B structured-JSON failures across repeated judge runs. V12 uses Vertex AI controlled generation with `responseMimeType=application/json` plus an explicit `responseSchema` for Gemini Judge B. The same schema mechanism is also used for Gemini adjudication and batched Gemini generation.

## Runtime improvements

- Initial generation changes from three Gemini calls per assignment to one structured batch call returning A/B/C. This reduces the normal initial generation call count by about 67 percent.
- Gemini Judge B no longer depends on repeated free-form JSON repair attempts. The JSON retry budget is reduced from two retries to one fallback retry.
- Judge B and adjudicator output budgets are reduced because their structured outputs are compact.
- Inter-call pacing is reduced from 0.8 seconds to 0.25 seconds.
- Existing two-judge concurrency is preserved. A configurable per-judge concurrency setting is included for future controlled scaling.

## Methodology preserved

- Gemini 3.1 Pro remains generator, repairer, refiller, and blind adjudicator.
- gpt-oss-120B and Gemini 2.5 Pro remain two independent judges.
- A technical judge failure is not a vote.
- Three, two, one, and zero valid-candidate recovery rules remain unchanged.
- Deterministic validation, repair, refill, adjudication, and final reporting remain separate stages.
- Test/pilot/production targets remain 9/100/500.
