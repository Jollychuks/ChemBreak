from __future__ import annotations
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from .base import TargetSession
from .hf_common import HFGeneratorMixin, dtype_from_name


class ChemDFMSession(TargetSession, HFGeneratorMixin):
    def __init__(self, spec, generation):
        super().__init__()
        repo = spec["repo_id"]
        self.tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=bool(spec.get("trust_remote_code", False)))
        self.model = AutoModelForCausalLM.from_pretrained(
            repo,
            torch_dtype=dtype_from_name(spec.get("torch_dtype", "bfloat16")),
            device_map="auto",
            trust_remote_code=bool(spec.get("trust_remote_code", False)),
        )
        self.model.eval()
        self.gen_cfg = GenerationConfig(
            do_sample=bool(generation.get("do_sample", False)),
            temperature=float(generation.get("temperature", 0.0)) if generation.get("do_sample", False) else None,
            max_new_tokens=int(generation.get("max_new_tokens", 768)),
            repetition_penalty=float(generation.get("repetition_penalty", 1.05)),
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def _format(self, query: str) -> str:
        text = ""
        for idx, item in enumerate(self.history):
            text += f"[Round {idx}]\nHuman: {item['user']}\nAssistant: {item['assistant']}\n"
        text += f"[Round {len(self.history)}]\nHuman: {query}\nAssistant:"
        return text

    def ask(self, query: str) -> str:
        response = self._generate_from_prompt(self._format(query))
        self.history.append({"user": query, "assistant": response})
        return response
