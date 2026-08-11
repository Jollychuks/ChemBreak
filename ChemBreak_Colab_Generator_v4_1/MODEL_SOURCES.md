# V4 Model Sources

The four family representatives are centralized in `model_registry.json`.

## Family A: Qwen

Model:
`Qwen/Qwen3.5-9B`

Official source:
https://huggingface.co/Qwen/Qwen3.5-9B

V4 use:
- post-trained instruction-capable model
- latest Transformers support
- text-only task in this experiment
- thinking disabled through the chat template
- 4-bit local loading for Colab memory control

## Family B: Mistral

Model:
`mistralai/Ministral-3-8B-Instruct-2512-BF16`

Official source:
https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16

Related official usage card:
https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512

V4 use:
- instruct post-trained model
- MistralCommonBackend tokenizer
- Mistral3ForConditionalGeneration
- 4-bit loading from the BF16 checkpoint

## Family C: Gemma

Model:
`google/gemma-4-E4B-it`

Official source:
https://huggingface.co/google/gemma-4-E4B-it

Official documentation:
https://ai.google.dev/gemma/docs/core

V4 use:
- instruction-tuned Gemma 4 E4B
- AutoProcessor + AutoModelForMultimodalLM
- text-only task in this experiment
- thinking disabled
- 4-bit local loading

## Family D: Phi

Model:
`microsoft/phi-4`

Official source:
https://huggingface.co/microsoft/phi-4

V4 use:
- 14B dense decoder-only text model
- AutoTokenizer + AutoModelForCausalLM
- chat template
- 4-bit local loading

## Why four different families

The purpose is not to claim that these four are universally the best open-weight models.

They are four distinct model lineages that are practical candidates for the ChemBreak comparison under local Colab constraints.

The model registry can be revised later without changing the ChemBreak matrix, taxonomy, scenario plan logic, generator prompt, judge rubric, or aggregation methodology.
