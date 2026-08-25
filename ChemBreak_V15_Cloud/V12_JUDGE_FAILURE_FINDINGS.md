# V12 Judge Failure Findings Used for V15

Observed V12 test behavior:

- Judge A and Judge B were launched concurrently.
- gpt-oss responses were often parseable JSON but used a score layout that omitted the V12 nested `scores` object.
- Gemini 2.5 Pro responses repeatedly failed JSON parsing with truncated or malformed output during the deeply nested judge response.
- V12 therefore marked assignments `TECHNICAL_PENDING` rather than treating technical failures as votes.
- Some gpt-oss retries lasted several minutes, making the failed judge stage unnecessarily slow.

V15 addresses the response contract and latency while preserving two independent judges.
