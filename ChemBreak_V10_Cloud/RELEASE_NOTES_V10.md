# ChemBreak V10 release notes

V10 is a fresh major-version namespace and GitHub folder.

## Included

- GitHub-connected Google Colab workflow
- Optional Colab Enterprise controller notebook
- Gemini 3.1 Pro Preview primary generation
- Llama 4 Maverick managed API generation
- gpt-oss-120B generation, repair, and independent reasoning judgment
- Gemini 2.5 Pro stable judgment
- Gemini 3.1 Pro Preview adjudication
- Retry/backoff for HTTP 429 and transient server failures
- Gemini thought-part handling
- gpt-oss reasoning-output handling
- deterministic validation with all-defect reporting
- within-bank similarity checks
- external HarmBench similarity reference
- dynamic opening-language diversity controls
- resumable CSV/JSONL checkpoints
- Google Drive persistence in standard Colab
- optional Cloud Storage synchronization
- V10 run signature in plan and final summary
- generation progress and ETA
- fresh V10 assignment IDs and output files

## V10 namespace

Assignment IDs use `CBV10C-####`.

V10 does not import or call any earlier ChemBreak code folder.


## Live progress output

V10 now provides visible progress throughout long runs:

- stage START and DONE banners
- per-model-call START/DONE/ERROR messages
- a heartbeat about every 20 seconds while waiting for a Vertex response
- completed/total counts
- percentage complete
- elapsed time
- ETA
- candidate IDs and model roles during generation
- PASS/FAIL and defect previews during validation
- per-repair status during repair
- candidate pairs and individual judge decisions during judging
- visible adjudication when judges disagree
- refill progress
- finalization progress

The progress output is flushed immediately so it appears continuously in Google Colab rather than only when a stage finishes.
