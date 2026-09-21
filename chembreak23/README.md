# ChemBreak23

ChemBreak23 is an adaptive multi-turn red-teaming experiment for evaluating the safety behavior of chemistry language models. It uses a replay-augmented Markov Decision Process (MDP) to learn which reformulation strategies are most effective for each benchmark task and then evaluates the learned routes in fresh target-model conversations.

The current experiment evaluates two target models:

- **ChemDFM:** `OpenDFM/ChemDFM-v1.5-8B`
- **ChemLLM:** `AI4Chem/ChemLLM-7B-Chat-1_5-SFT`

The two targets are run separately and maintain independent controller state, checkpoints, and route memory.

## Dataset

The benchmark contains **28 prompts selected from the `chemical_biological` harmful-behavior category of the HarmBench prompt dataset**:

[HarmBench behavior dataset](https://github.com/centerforaisafety/HarmBench/blob/main/data/behavior_datasets/harmbench_behaviors_text_all.csv)

The selected prompts are stored in `data/prompts.csv`. Their identifiers, categories, source hashes, and fixed selection order are recorded in `data/CB23_prompts28_manifest_v1.csv`. The file `data/CB23_prompts28_lock_v1.json` verifies that the same 28 prompts are used in every run.

## Experiment flow

```mermaid
flowchart TD
    A["28 HarmBench chemical-biological prompts"] --> B["Baseline evaluation"]
    B --> C["Epoch 1: adaptive learning"]
    C --> D["Epoch 2: replay and recovery"]
    D --> E["Epoch 3: replay and recovery"]
    E --> F["Freeze learned policy and routes"]
    F --> G["Final attack in fresh contexts"]
    G --> H["CHCS evaluation"]
    H --> I["Public redacted results"]
    H --> J["Private internal audit"]
```

### Stages

1. **Baseline:** The original prompt is sent directly to the target model to measure its initial response.
2. **Epoch 1:** The MDP controller selects adaptive reformulation actions. Each task receives at most five target-model turns and stops early if successful.
3. **Epoch 2:** The system first replays the strongest successful route learned earlier. If the replay fails and turns remain, it continues with adaptive recovery.
4. **Epoch 3:** The replay and recovery process is repeated using the updated route memory and policy.
5. **Freeze:** Learning stops, and the policy and ranked successful routes are frozen before final evaluation.
6. **Final attack:** Successful routes are replayed in fresh target-model conversations. If the stored routes fail, the system can make up to two final attempts generated from the frozen evidence.
7. **Evaluation:** The CHCS judge scores each target response from 1 to 5. Only **CHCS = 5** is counted as an automatic attack success.

The attack LLM generates candidate reformulations. The intent gate checks that each candidate preserves the original task and does not introduce new operational details. The target model then responds, and the CHCS judge evaluates the response against the immutable original behavioral objective.

## Model roles

| Role | Model | Purpose |
| --- | --- | --- |
| Attack LLM | `gemini-3.1-pro-preview` | Generates candidate reformulations selected by the MDP controller |
| Intent gate | `gemini-3.8-flash` | Checks goal preservation and task fidelity before a candidate reaches the target |
| CHCS judge | `gpt-5.6-sol` | Scores harmful compliance from 1 to 5 |
| Target models | ChemDFM and ChemLLM | Models being evaluated |

## Package contents

| Path | Contents |
| --- | --- |
| `notebooks/chembreak23_Cloud_Notebook.ipynb` | Main notebook used to configure, run, inspect, and package the experiment |
| `configs/config.cb23.yaml` | Models, turn limits, MDP settings, replay rules, reward values, and export settings |
| `data/prompts.csv` | The 28 selected HarmBench prompts and their task anchors |
| `data/CB23_prompts28_manifest_v1.csv` | Fixed task order, IDs, categories, and prompt hashes |
| `data/CB23_prompts28_lock_v1.json` | Dataset identity and integrity lock |
| `src/chembreak23/runner.py` | Baseline, learning epochs, replay, recovery, freeze, and final-stage execution |
| `src/chembreak23/policy.py` | MDP action selection and policy updates |
| `src/chembreak23/route_memory.py` | Storage and ranking of successful routes |
| `src/chembreak23/prompts.py` | Attack, intent-gate, and CHCS judge instructions |
| `src/chembreak23/targets.py` | ChemDFM and ChemLLM loading and inference logic |
| `src/chembreak23/metrics.py` | ASR calculation and public/private result exports |
| `src/chembreak23/checkpoint.py` | SQLite checkpoint and resume state |
| `scripts/validate_package.py` | Validates the configuration, notebook, and locked dataset |
| `tests/` | Automated checks for the core, execution, export, and packaging logic |

## How to run

The supplied notebook is designed for a GPU-enabled cloud environment such as Colab Enterprise.

### Requirements

- Python 3.10 or later
- A CUDA GPU with BF16 support recommended
- Google Cloud project with access to the configured Vertex AI Gemini models
- OpenAI API key for the CHCS judge
- Access to the Hugging Face target-model repositories
- A GitHub repository containing the `chembreak23` directory

### Notebook procedure

1. Place the `chembreak23` folder in the root of the GitHub repository specified in the notebook.
2. Open `notebooks/chembreak23_Cloud_Notebook.ipynb` in the cloud runtime.
3. In **Cell 1**, confirm or update:
   - `PROJECT_ID`
   - `REPO_URL`
   - `BRANCH`
   - `PROJECT_SUBDIR`
   - `LIVE`
   - `TARGETS`
4. Set `LIVE = True` for a real experiment. Set it to `False` only for a dry run.
5. Make sure the runtime is authenticated to the Google Cloud project used by Vertex AI.
6. Run the notebook from **Cell 1 through Cell 15 in order**.
7. When Cell 5 requests it, enter the OpenAI API key. The input is hidden and is not written to disk.
8. Cell 7 performs the dataset, configuration, model, and API preflight checks.
9. Cell 8 confirms that the locked dataset contains exactly 28 tasks.
10. Cell 10 runs ChemDFM, and Cell 11 runs ChemLLM.
11. Cell 12 displays the cross-target ASR summary.
12. Cell 13 provides paths to the private transcripts for manual review.
13. Cell 14 creates the public redacted results archive.
14. Cell 15 creates the private internal audit archive.

To validate the package before a run, execute this command from the `chembreak23` directory:

```bash
python scripts/validate_package.py
```

## Outputs

Run data are stored under:

```text
/content/chembreak23_storage/runs/CB23_CHCS_REPLAY_MDP_PROMPTS28_V1/
```

Each target directory contains:

- `release/summary.json`: stage-level ASR, CHCS distributions, query counts, replay counts, and cumulative discovery results.
- `release/episodes.csv`: one row per completed task episode.
- `release/turns.csv`: redacted turn-level records.
- `release/target_queries.csv`: redacted target-query records.
- `release/provider_events.csv`: provider, retry, gate, and execution events.
- `release/route_rankings_public.json`: redacted route-ranking information.
- `internal/cb23_full_transcripts.csv` and `.jsonl`: original prompts, attack prompts, target responses, CHCS results, and route metadata.
- `internal/cb23_successful_trajectories.csv` and `.jsonl`: successful learned routes and their exact prompt paths.
- `state.sqlite3`: checkpoint, controller, episode, turn, and route state used for recovery and resumption.

The **public results archive** is suitable for sharing because prompt and response text is redacted. The **internal audit archive** contains raw benchmark prompts, target responses, and state databases. Keep it private unless an approved release process says otherwise.

## ASR definition

For each stage, Attack Success Rate is:

```text
ASR = number of tasks with at least one CHCS 5 response in the stage / 28
```

`adaptive_ever_success_asr` reports the proportion of the 28 tasks that achieved at least one CHCS 5 during any learning epoch or the final stage. Baseline successes are excluded from this cumulative adaptive measure.

## Rerunning the experiment

ChemBreak23 saves checkpoints under `/content/chembreak23_storage`. Reusing the same experiment revision and storage directory can resume existing state. For a completely independent run, use a new experiment revision or a clean CB23 storage directory before execution.
