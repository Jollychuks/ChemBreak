from __future__ import annotations
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


def dtype_from_name(name: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(name, torch.bfloat16)


class HFGeneratorMixin:
    def _generate_from_prompt(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        gen = self.gen_cfg
        with torch.inference_mode():
            out = self.model.generate(**inputs, generation_config=gen)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def close(self) -> None:
        try:
            del self.model
            del self.tokenizer
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
