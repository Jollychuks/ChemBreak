from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

from chembreak_common import gpu_report


@dataclass
class LoadedModel:
    family_id: str
    family_config: Dict[str, Any]
    model: Any
    processor: Any


def _bnb_config() -> BitsAndBytesConfig:
    compute_dtype = (
        torch.bfloat16
        if (
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )
        else torch.float16
    )

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _input_device(model: Any) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        pass

    try:
        return next(
            p.device
            for p in model.parameters()
            if p.device.type != "meta"
        )
    except Exception:
        return torch.device("cuda:0")


def _text_messages(
    family: Dict[str, Any],
    system_message: str,
    user_prompt: str,
):
    style = family.get(
        "message_content_style",
        "plain",
    )

    if style == "text_blocks":
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_message,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    }
                ],
            },
        ]

    return [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def load_family_model(
    family_id: str,
    registry: Dict[str, Any],
    config: Dict[str, Any],
    hf_token: Optional[str] = None,
) -> LoadedModel:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. In Colab "
            "choose Runtime > Change runtime type > GPU."
        )

    family = registry["families"][family_id]
    model_id = family["model_id"]
    loader_kind = family["loader_kind"]
    cache_dir = config.get("hf_cache_dir") or None

    print(
        f"\nLoading family {family_id}: "
        f"{family['family_name']}"
    )
    print("Model:", model_id)
    print("GPU:", gpu_report())

    kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "cache_dir": cache_dir,
        "low_cpu_mem_usage": True,
        "trust_remote_code": bool(
            family.get(
                "trust_remote_code",
                False,
            )
        ),
    }

    if hf_token:
        kwargs["token"] = hf_token

    if bool(
        family.get(
            "load_in_4bit",
            True,
        )
    ):
        kwargs["quantization_config"] = (
            _bnb_config()
        )
    else:
        kwargs["torch_dtype"] = "auto"

    processor_kwargs: Dict[str, Any] = {
        "cache_dir": cache_dir,
        "trust_remote_code": bool(
            family.get(
                "trust_remote_code",
                False,
            )
        ),
    }

    if hf_token:
        processor_kwargs["token"] = hf_token

    if loader_kind == "causal_lm":
        processor = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            **processor_kwargs,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **kwargs,
        )

    elif loader_kind == "auto_multimodal":
        try:
            from transformers import (
                AutoModelForMultimodalLM,
            )
        except ImportError as exc:
            raise RuntimeError(
                "AutoModelForMultimodalLM is "
                "unavailable. Rerun the V4 "
                "dependency cell, which installs "
                "the latest Transformers."
            ) from exc

        processor = AutoProcessor.from_pretrained(
            model_id,
            **processor_kwargs,
        )

        model = (
            AutoModelForMultimodalLM
            .from_pretrained(
                model_id,
                **kwargs,
            )
        )

    elif loader_kind == "mistral3":
        try:
            from transformers import (
                Mistral3ForConditionalGeneration,
                MistralCommonBackend,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Ministral 3 support is "
                "unavailable. Rerun the V4 "
                "dependency cell and confirm "
                "mistral-common is installed."
            ) from exc

        mistral_kwargs: Dict[str, Any] = {}
        if hf_token:
            mistral_kwargs["token"] = hf_token

        processor = (
            MistralCommonBackend
            .from_pretrained(
                model_id,
                **mistral_kwargs,
            )
        )

        model = (
            Mistral3ForConditionalGeneration
            .from_pretrained(
                model_id,
                **kwargs,
            )
        )

    else:
        raise ValueError(
            f"Unknown loader_kind: "
            f"{loader_kind}"
        )

    model.eval()

    if hasattr(
        processor,
        "pad_token_id",
    ):
        if (
            processor.pad_token_id is None
            and getattr(
                processor,
                "eos_token_id",
                None,
            ) is not None
        ):
            processor.pad_token_id = (
                processor.eos_token_id
            )

    if hasattr(
        processor,
        "tokenizer",
    ):
        tok = processor.tokenizer
        if (
            getattr(
                tok,
                "pad_token_id",
                None,
            ) is None
            and getattr(
                tok,
                "eos_token_id",
                None,
            ) is not None
        ):
            tok.pad_token_id = tok.eos_token_id

    print("Model loaded.")

    return LoadedModel(
        family_id=family_id,
        family_config=family,
        model=model,
        processor=processor,
    )


def _prepare_inputs(
    loaded: LoadedModel,
    system_message: str,
    user_prompt: str,
):
    family = loaded.family_config
    processor = loaded.processor
    loader_kind = family["loader_kind"]

    messages = _text_messages(
        family,
        system_message,
        user_prompt,
    )

    if loader_kind == "mistral3":
        inputs = processor.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        template_kwargs: Dict[str, Any] = {}

        if (
            family.get(
                "enable_thinking"
            ) is not None
        ):
            template_kwargs[
                "enable_thinking"
            ] = bool(
                family[
                    "enable_thinking"
                ]
            )

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        )

    device = _input_device(
        loaded.model
    )

    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        inputs = {
            k: (
                v.to(device)
                if hasattr(v, "to")
                else v
            )
            for k, v in inputs.items()
        }

    return inputs


def generate_chat(
    loaded: LoadedModel,
    *,
    system_message: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    seed: int,
) -> str:
    set_seed(int(seed))

    processor = loaded.processor
    model = loaded.model

    inputs = _prepare_inputs(
        loaded,
        system_message,
        user_prompt,
    )

    input_len = (
        inputs["input_ids"].shape[-1]
    )

    do_sample = (
        float(temperature) > 0
    )

    gen_kwargs: Dict[str, Any] = {
        **inputs,
        "max_new_tokens":
            int(max_new_tokens),
        "do_sample":
            do_sample,
        "repetition_penalty":
            float(repetition_penalty),
    }

    if do_sample:
        gen_kwargs["temperature"] = (
            float(temperature)
        )
        gen_kwargs["top_p"] = (
            float(top_p)
        )

    pad_id = getattr(
        processor,
        "pad_token_id",
        None,
    )
    eos_id = getattr(
        processor,
        "eos_token_id",
        None,
    )

    if hasattr(
        processor,
        "tokenizer",
    ):
        pad_id = getattr(
            processor.tokenizer,
            "pad_token_id",
            pad_id,
        )
        eos_id = getattr(
            processor.tokenizer,
            "eos_token_id",
            eos_id,
        )

    if pad_id is not None:
        gen_kwargs[
            "pad_token_id"
        ] = pad_id

    if eos_id is not None:
        gen_kwargs[
            "eos_token_id"
        ] = eos_id

    with torch.inference_mode():
        outputs = model.generate(
            **gen_kwargs
        )

    generated = outputs[0][
        input_len:
    ]

    if (
        loaded.family_config[
            "loader_kind"
        ] == "mistral3"
    ):
        # MistralCommonBackend's documented decode
        # call does not require the Hugging Face
        # skip_special_tokens keyword.
        text = processor.decode(
            generated
        )
    elif hasattr(
        processor,
        "decode",
    ):
        text = processor.decode(
            generated,
            skip_special_tokens=True,
        )
    elif hasattr(
        processor,
        "tokenizer",
    ):
        text = (
            processor.tokenizer
            .decode(
                generated,
                skip_special_tokens=True,
            )
        )
    else:
        raise RuntimeError(
            "Loaded processor/tokenizer "
            "does not provide decode()."
        )

    return str(text).strip()


def unload_family_model(
    loaded: Optional[LoadedModel],
) -> None:
    if loaded is None:
        return

    try:
        del loaded.model
    except Exception:
        pass

    try:
        del loaded.processor
    except Exception:
        pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
