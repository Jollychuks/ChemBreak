# ChemBreak Colab Generator v1

This package is the no-paid-LLM-API version of the ChemBreak candidate-task generator.

The model runs directly inside the Google Colab runtime through Hugging Face Transformers.

## Default model

`Qwen/Qwen3-4B-Instruct-2507`

This is a public Qwen model that can be loaded directly with Transformers. The package does not contain model weights. Colab downloads the model from Hugging Face when you load it.

Official model page:
https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507

The default configuration uses 4-bit bitsandbytes quantization to reduce GPU-memory requirements.

Official quantization documentation:
https://huggingface.co/docs/transformers/quantization/bitsandbytes

Google Colab resource limits are dynamic and GPUs are not guaranteed on the free tier:
https://research.google.com/colaboratory/faq.html

## No paid API key

There is no OpenAI, Anthropic, Gemini, or other paid LLM API call in this package.

The notebook calls:

`model.generate(...)`

inside Colab.

The chosen default model is public, so the normal workflow does not require a Hugging Face access token either. If you later switch to a gated Hugging Face model, that specific model may require a free Hugging Face account/token.

## Files

### `ChemBreak_Colab_Generator.ipynb`
Main Google Colab notebook. Start here.

### `generate_local.py`
Local Hugging Face generation engine. It loads the model, loops through the matrix, validates JSON, checkpoints output, and resumes completed work.

### `ChemBreak_Working_Generation_Matrix_v1.csv`
The 271-row generation matrix.

### `generator_prompt.txt`
Reusable ChemBreak generation prompt.

### `run_config.json`
Controls model, number of candidates, selected rows, generation settings, output paths, and resume behavior.

### `candidate_tasks_template.csv`
Shows the schema of the generated task bank.

### `requirements_colab.txt`
Python packages installed by the notebook.

## One-time Google Drive setup

1. Download and extract `ChemBreak_Colab_Generator_v1.zip`.
2. Upload the entire extracted folder to the root of Google Drive `My Drive`.
3. The folder should be:

`MyDrive/ChemBreak_Colab_Generator_v1/`

4. Open `ChemBreak_Colab_Generator.ipynb` with Google Colab.
5. In Colab choose `Runtime > Change runtime type > GPU`.
6. Run the cells from top to bottom.

The notebook mounts Google Drive and sets:

`PROJECT_DIR = /content/drive/MyDrive/ChemBreak_Colab_Generator_v1`

If you put the folder somewhere else, change `PROJECT_DIR` in the notebook.

## Default first run

The supplied `run_config.json` is intentionally a pilot:

- `start_row`: 1
- `end_row`: 10
- `n_per_row`: 5
- `call_batch_size`: 5

That targets:

10 matrix rows × 5 candidates = 50 candidates.

Inspect those 50 before scaling.

## Full run

After the pilot looks right, edit `run_config.json`:

```json
{
  "fit": "ALL",
  "n_per_row": 25,
  "start_row": 1,
  "end_row": 271
}
```

With 271 rows, that targets:

271 × 25 = 6,775 raw candidate tasks.

For 50 per row:

271 × 50 = 13,550 raw candidates.

These are raw candidates, not automatically accepted benchmark tasks.

## Resume/checkpoint behavior

`resume` is `true` by default.

Each accepted local generation batch is immediately appended to:

`candidate_tasks.csv`

in your Google Drive project folder.

Candidate IDs look like:

- `GM001-C0001`
- `GM001-C0002`
- `GM002-C0001`

When Colab disconnects, reconnect and rerun the notebook. Existing candidate IDs are skipped.

The generator also writes:

- `generation_progress.csv`
- `generation_errors.jsonl`

This means a long run can continue without starting over.

## Matrix subsets

### CORE only

Set:

```json
"fit": "CORE"
```

### CORE + SELECTIVE

Set:

```json
"fit": "ALL"
```

### Particular matrix rows

Set:

```json
"matrix_ids": ["GM001", "GM017", "GM104"]
```

When `matrix_ids` is non-empty, only those IDs are selected before the start/end slice.

## Candidate batching

`call_batch_size` controls how many candidates the local model is asked to emit in one JSON response.

Recommended starting value:

```json
"call_batch_size": 5
```

For a 25-candidate row, the script makes 5 local generation calls of 5 candidates each.

This does not incur an API charge. It only uses Colab compute.

## 4-bit mode

Default:

```json
"load_in_4bit": true
```

This uses bitsandbytes quantization so the 4B model is more practical on a Colab GPU.

If you get an out-of-memory error:

1. Confirm 4-bit mode is true.
2. Reduce `call_batch_size`.
3. Reduce `max_new_tokens`.
4. Use a smaller public model.
5. Restart the Colab runtime to clear GPU memory.

## Model cache

Default:

`/content/hf_cache`

This is fast because it is on the Colab runtime disk, but it disappears when the runtime is destroyed.

Your generated CSV and progress files are stored in Google Drive and persist.

You can change `hf_cache_dir` to a Drive path if you want to persist downloaded model files, but Drive-backed model loading can be slower.

## Outputs

`candidate_tasks.csv` contains:

- candidate_id
- matrix_id
- candidate_index
- hc_id
- hc_category
- hd_id
- hazard_domain
- fit
- ot_id
- output_type
- allowed_scenarios
- selected_scenarios
- benchmark_prompt
- main_goal
- chemical_entity
- distinctive_dimension
- generator_model
- prompt_version
- generation_seed
- generated_at_utc

## Recommended research workflow

1. Generate 50 pilot candidates.
2. Manually inspect category fit and task quality.
3. Adjust `generator_prompt.txt` if needed.
4. Generate 500 to check diversity.
5. Scale into thousands.
6. Run the separate ChemBreak filtering and deduplication pipeline.
7. Human-review the retained benchmark set.
8. Freeze a benchmark release/version.

The generator is only the candidate-authoring stage.
