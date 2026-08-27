from __future__ import annotations
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
from .base import TargetSession
from .hf_common import HFGeneratorMixin, dtype_from_name


class LlaSMolSession(TargetSession, HFGeneratorMixin):
    def __init__(self, spec, generation):
        super().__init__()
        base_id = spec["base_model_id"]
        adapter_id = spec["adapter_id"]
        self.tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=bool(spec.get("trust_remote_code", False)))
        base = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=dtype_from_name(spec.get("torch_dtype", "bfloat16")),
            device_map="auto",
            trust_remote_code=bool(spec.get("trust_remote_code", False)),
        )
        self.model = PeftModel.from_pretrained(base, adapter_id)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gen_cfg = GenerationConfig(
            do_sample=bool(generation.get("do_sample", False)),
            temperature=float(generation.get("temperature", 0.0)) if generation.get("do_sample", False) else None,
            max_new_tokens=int(generation.get("max_new_tokens", 768)),
            repetition_penalty=float(generation.get("repetition_penalty", 1.05)),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def _format(self, query: str) -> str:
        if not self.history:
            return f"Query: {query}\nResponse:"
        text = "Previous conversation:\n"
        for item in self.history:
            text += f"Query: {item['user']}\nResponse: {item['assistant']}\n"
        text += f"Query: {query}\nResponse:"
        return text

    def ask(self, query: str) -> str:
        response = self._generate_from_prompt(self._format(query))
        self.history.append({"user": query, "assistant": response})
        return response
