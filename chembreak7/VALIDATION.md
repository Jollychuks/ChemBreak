# Validation record

ChemBreak7 validation is performed against the standalone package before archiving.

Required checks:

- Frozen task bank hash and 500-row contract.
- Disjoint 8-task development, 40-task pilot, and 452-task holdout partitions.
- Adaptive-only configuration and source tree.
- Strict response schemas for all model roles.
- Audited Llama maximum-score normalization from 5 to the registered maximum 4.
- Unexpected out-of-range verifier scores remain errors.
- Bootstrap turn makes no actor call.
- Exactly one combined actor call on each later turn.
- No stagnation termination.
- Observer cannot declare success.
- Full hard gates are required for success.
- Transcript, observation, and evaluation checkpoint order.
- Pending-stage recovery without another target query.
- Three independent target cells sharing one signed run.
- Full 24-episode development mock.
- Resume without duplicate episodes, turns, observations, evaluations, or API calls.
- Notebook syntax and dependency-install safety.
- Archive integrity and absence of runtime caches.

## Release result

- Automated tests: 24 passed.
- Static lint checks: passed.
- Python and notebook syntax checks: passed.
- Mock development episodes: 24 complete out of 24.
- Mock target transcripts: 120.
- Structured observations: 120.
- Saved evaluations: 120.
- Combined adaptive actor calls: 96.
- Full safety-verifier calls: 48.
- Full chemistry-verifier calls: 48.
- Historical failure events: 0.
- Resume pass: no added episodes, transcripts, observations, evaluations, or API calls.
- Frozen task-bank SHA-256: `62df773ce8c4a252bd23350fcd3a8e83fc670864efb0304fe8c65d21d3c7d6ff`.

Authenticated live provider calls and multi-gigabyte target-weight loading require the user's Google Cloud project and are intentionally not represented as local validation.
