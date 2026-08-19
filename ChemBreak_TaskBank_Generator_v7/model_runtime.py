"""Checkpoint-specific runtime for the three ChemBreak V7 model families.

The public notebook remains simple. This module contains the unavoidable
differences between the current Qwen, Ministral, and Phi-4 checkpoints.
Only one checkpoint is kept in GPU memory at a time.
"""

from __future__ import annotations

import gc
import os
from importlib.metadata import PackageNotFoundError, version
from typing import Any


MINIMUM_TRANSFORMERS_VERSION = "5.15.0"


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def runtime_preflight(registry: dict[str, Any]) -> dict[str, Any]:
    """Verify the GPU stack and every required loader before model downloads."""
    try:
        import accelerate
        import bitsandbytes
        import torch
        import transformers
        from huggingface_hub import model_info
        from packaging.version import Version
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForMultimodalLM,
            AutoProcessor,
            AutoTokenizer,
            Mistral3ForConditionalGeneration,
            MistralCommonBackend,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The V7 model runtime is incomplete. Run the dependency cell, then "
            "rerun the preflight cell before starting ChemBreak."
        ) from exc

    # Keep these references explicit so a missing lazy export fails here instead
    # of after a multi-gigabyte checkpoint download.
    required_classes = {
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForMultimodalLM": AutoModelForMultimodalLM,
        "AutoProcessor": AutoProcessor,
        "AutoTokenizer": AutoTokenizer,
        "Mistral3ForConditionalGeneration": Mistral3ForConditionalGeneration,
        "MistralCommonBackend": MistralCommonBackend,
    }
    if any(value is None for value in required_classes.values()):
        raise RuntimeError("One or more required Transformers loaders are unavailable.")

    if Version(transformers.__version__) < Version(MINIMUM_TRANSFORMERS_VERSION):
        raise RuntimeError(
            f"Transformers {MINIMUM_TRANSFORMERS_VERSION} or newer is required; "
            f"found {transformers.__version__}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU is available. In Colab choose Runtime > Change runtime "
            "type > A100, L4, or T4 GPU, then rerun from the dependency cell."
        )

    expected_loaders = {
        "A": "auto_multimodal",
        "B": "mistral3",
        "C": "causal_lm",
    }
    expected_roles = {"A": "generator", "B": "generator", "C": "judge"}
    families = registry.get("families", {})
    repository_access: dict[str, dict[str, Any]] = {}
    hub_token = os.environ.get("HF_TOKEN") or None
    for family_id, loader_kind in expected_loaders.items():
        if family_id not in families:
            raise ValueError(f"models.json is missing family {family_id}.")
        actual = str(families[family_id].get("loader_kind", ""))
        if actual != loader_kind:
            raise ValueError(
                f"Family {family_id} must use loader_kind={loader_kind!r}, "
                f"not {actual!r}."
            )
        actual_role = str(families[family_id].get("role", ""))
        if actual_role != expected_roles[family_id]:
            raise ValueError(
                f"Family {family_id} must use role={expected_roles[family_id]!r}, "
                f"not {actual_role!r}."
            )
        model_id = str(families[family_id]["model_id"])
        try:
            details = model_info(model_id, token=hub_token)
        except Exception as exc:
            raise RuntimeError(
                f"The checkpoint {model_id} could not be reached. Check the model "
                "ID, network, and optional HF_TOKEN, then rerun the "
                "preflight before downloading weights."
            ) from exc
        gated = getattr(details, "gated", False)
        private = bool(getattr(details, "private", False))
        if private or gated not in (False, None, "false"):
            raise RuntimeError(
                f"{model_id} no longer appears to be a public, ungated checkpoint. "
                "Stop and review the model registry before running V7."
            )
        repository_access[family_id] = {
            "model_id": model_id,
            "public": not private,
            "gated": bool(gated),
        }

    properties = torch.cuda.get_device_properties(0)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gb": round(properties.total_memory / (1024**3), 2),
        "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "huggingface_hub": _package_version("huggingface-hub"),
        "mistral_common": _package_version("mistral-common"),
        "models": {
            family_id: families[family_id]["model_id"]
            for family_id in ("A", "B", "C")
        },
        "repository_access": repository_access,
    }


class ModelRunner:
    """Load one configured checkpoint, generate text, and release GPU memory."""

    def __init__(self, family_id: str, model_info: dict[str, Any]) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForMultimodalLM,
                AutoProcessor,
                AutoTokenizer,
                BitsAndBytesConfig,
                Mistral3ForConditionalGeneration,
                MistralCommonBackend,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Install requirements_colab.txt and pass the V7 preflight before "
                "loading a model."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required for the three real checkpoints.")

        self.torch = torch
        self.family_id = family_id
        self.model_info = model_info
        self.model_id = str(model_info["model_id"])
        self.loader_kind = str(model_info["loader_kind"])
        self.processor: Any | None = None
        self.model: Any | None = None

        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        quantization = BitsAndBytesConfig(
            load_in_4bit=bool(model_info.get("load_in_4bit", True)),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        cache_dir = os.environ.get("HF_HOME") or None
        hub_token = os.environ.get("HF_TOKEN") or None
        common_model_kwargs: dict[str, Any] = {
            "cache_dir": cache_dir,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": bool(model_info.get("trust_remote_code", False)),
            "dtype": compute_dtype,
            "quantization_config": quantization,
            "token": hub_token,
        }
        common_processor_kwargs: dict[str, Any] = {
            "cache_dir": cache_dir,
            "trust_remote_code": bool(model_info.get("trust_remote_code", False)),
            "token": hub_token,
        }

        try:
            if self.loader_kind == "auto_multimodal":
                self.processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    **common_processor_kwargs,
                )
                self.model = AutoModelForMultimodalLM.from_pretrained(
                    self.model_id,
                    **common_model_kwargs,
                )
            elif self.loader_kind == "mistral3":
                mistral_processor_kwargs: dict[str, Any] = {"token": hub_token}
                self.processor = MistralCommonBackend.from_pretrained(
                    self.model_id,
                    **mistral_processor_kwargs,
                )
                self.model = Mistral3ForConditionalGeneration.from_pretrained(
                    self.model_id,
                    **common_model_kwargs,
                )
            elif self.loader_kind == "causal_lm":
                self.processor = AutoTokenizer.from_pretrained(
                    self.model_id,
                    **common_processor_kwargs,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    **common_model_kwargs,
                )
            else:
                raise ValueError(f"Unsupported loader_kind: {self.loader_kind}")
        except Exception as exc:
            self.close()
            raise RuntimeError(
                f"Could not load {self.model_id} with the V7 "
                f"{self.loader_kind} loader. Confirm the dependency cell and GPU "
                "preflight succeeded, then rerun to resume."
            ) from exc

        self.model.eval()
        self._set_padding_token()

    def _set_padding_token(self) -> None:
        for tokenizer in (
            self.processor,
            getattr(self.processor, "tokenizer", None),
        ):
            if tokenizer is None:
                continue
            if (
                getattr(tokenizer, "pad_token_id", None) is None
                and getattr(tokenizer, "eos_token_id", None) is not None
            ):
                tokenizer.pad_token_id = tokenizer.eos_token_id

    def _messages(self, system: str, prompt: str) -> list[dict[str, Any]]:
        if self.model_info.get("message_content_style") == "text_blocks":
            return [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                },
            ]
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

    def _input_device(self) -> Any:
        try:
            return self.model.get_input_embeddings().weight.device
        except Exception:
            for parameter in self.model.parameters():
                if parameter.device.type != "meta":
                    return parameter.device
        return self.torch.device("cuda:0")

    def _prepare_inputs(self, system: str, prompt: str) -> Any:
        messages = self._messages(system, prompt)
        if self.loader_kind == "mistral3":
            inputs = self.processor.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            template_kwargs: dict[str, Any] = {}
            if self.model_info.get("enable_thinking") is not None:
                template_kwargs["enable_thinking"] = bool(
                    self.model_info["enable_thinking"]
                )
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **template_kwargs,
            )

        device = self._input_device()
        if hasattr(inputs, "to"):
            return inputs.to(device)
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _decode(self, generated: Any) -> str:
        if self.loader_kind == "mistral3":
            return str(self.processor.decode(generated)).strip()
        if hasattr(self.processor, "decode"):
            return str(
                self.processor.decode(generated, skip_special_tokens=True)
            ).strip()
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            return str(tokenizer.decode(generated, skip_special_tokens=True)).strip()
        raise RuntimeError(f"{self.model_id} has no supported decode method.")

    def generate(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        max_new_tokens: int,
        seed: int,
    ) -> str:
        self.torch.manual_seed(int(seed))
        self.torch.cuda.manual_seed_all(int(seed))
        inputs = self._prepare_inputs(system, prompt)
        input_length = int(inputs["input_ids"].shape[-1])
        do_sample = float(temperature) > 0
        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
            "repetition_penalty": float(repetition_penalty),
        }
        if do_sample:
            generation_kwargs.update(
                {"temperature": float(temperature), "top_p": float(top_p)}
            )

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id

        with self.torch.inference_mode():
            outputs = self.model.generate(**generation_kwargs)
        generated = outputs[0][input_length:]
        return self._decode(generated)

    def close(self) -> None:
        if getattr(self, "model", None) is not None:
            del self.model
            self.model = None
        if getattr(self, "processor", None) is not None:
            del self.processor
            self.processor = None
        gc.collect()
        torch_module = getattr(self, "torch", None)
        if torch_module is not None and torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
            try:
                torch_module.cuda.ipc_collect()
            except Exception:
                pass
