# V13 Test Findings Used for V15

The completed V13 test selected 9/9 assignments and produced 18 judgment records, 9 judge outcomes, 3 adjudications, and a final task bank of 9 rows.

The initial generation stage completed only 5/9 assignments in its first batch pass, producing 15 initial candidates instead of 27. Four assignments failed because Gemini 3.1 returned malformed structured JSON. Later refill recovered the missing assignments.

Initial deterministic validation passed 14/15 candidates; the single initial failure was repaired successfully. Judge agreement was 6/9 (0.667), with three genuine disagreements resolved by blind adjudication. The final maximum within-bank similarity was 0.1587.

The V13 final bank also exposed an entity-domain mapping issue: a chemical-warfare entity could enter HD3 because the source group `Chemical Weapons and Poisons` had been mapped broadly to `HD2|HD3`. V15 replaces this with stronger entity-specific mapping and validation.

These V13 outputs are findings only. V15 does not use V13 generated tasks as generation input.
