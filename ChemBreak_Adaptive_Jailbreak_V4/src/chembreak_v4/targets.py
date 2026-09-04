from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TargetResponse:
    text: str
    latency_seconds: float


class MockTarget:
    def __init__(self, target_id: str):
        self.target_id = target_id

    def load(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def generate(self, prompt: str, history: list[dict[str, str]]) -> TargetResponse:
        return TargetResponse(
            text="I cannot help with harmful chemical instructions. I can discuss general safety and emergency response.",
            latency_seconds=0.001,
        )


class HuggingFaceTarget:
    def __init__(self, settings: dict[str, Any]):
        self.settings = settings
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_name = self.settings.get("dtype", "bfloat16")
        dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
        model_name = self.settings["model"]
        trust = bool(self.settings.get("trust_remote_code", False))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=trust,
            low_cpu_mem_usage=True,
        )
        if self.settings["backend"] == "hf_peft":
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.settings["adapter"])
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def _format(self, prompt: str, history: list[dict[str, str]]) -> str:
        template = self.settings.get("template", "chat_template")
        if template == "chemdfm":
            pairs: list[tuple[str, str]] = []
            pending_user: str | None = None
            for message in history:
                if message["role"] == "user":
                    pending_user = message["content"]
                elif message["role"] == "assistant" and pending_user is not None:
                    pairs.append((pending_user, message["content"]))
                    pending_user = None
            chunks = [
                f"[Round {idx}]\nHuman: {user}\nAssistant: {assistant}\n"
                for idx, (user, assistant) in enumerate(pairs)
            ]
            chunks.append(f"[Round {len(pairs)}]\nHuman: {prompt}\nAssistant:")
            return "".join(chunks)
        messages = list(history) + [{"role": "user", "content": prompt}]
        if template == "mistral_instruct":
            pieces: list[str] = []
            for message in messages:
                if message["role"] == "user":
                    pieces.append(f"<s>[INST] {message['content']} [/INST]")
                else:
                    pieces.append(f" {message['content']}</s>")
            return "".join(pieces)
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return "\n".join(f"{m['role'].title()}: {m['content']}" for m in messages) + "\nAssistant:"

    def generate(self, prompt: str, history: list[dict[str, str]]) -> TargetResponse:
        import torch

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Target model is not loaded.")
        formatted = self._format(prompt, history)
        encoded = self.tokenizer(formatted, return_tensors="pt")
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        temperature = float(self.settings.get("temperature", 0.0))
        kwargs = {
            "max_new_tokens": int(self.settings.get("max_new_tokens", 512)),
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = float(self.settings.get("top_p", 0.9))
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**encoded, **kwargs)
        latency = time.perf_counter() - started
        new_tokens = output[0, encoded["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return TargetResponse(text=text, latency_seconds=latency)


def make_target(settings: dict[str, Any], dry_run: bool):
    if dry_run:
        return MockTarget(settings["id"])
    if settings["backend"] in {"hf_local", "hf_peft"}:
        return HuggingFaceTarget(settings)
    raise ValueError(f"Unsupported target backend: {settings['backend']}")

