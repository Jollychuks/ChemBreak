from __future__ import annotations
import gc, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class TargetResponse:
    text: str
    latency_seconds: float

class MockTarget:
    def __init__(self,target_id='ChemDFM'): self.target_id=target_id
    def load(self): return None
    def unload(self): return None
    def load_tokenizer_only(self): return {'target_id':self.target_id,'status':'mock'}
    def generate(self,prompt,history): return TargetResponse('I cannot provide harmful chemical instructions. I can discuss safety at a high level.',0.001)

class HuggingFaceTarget:
    def __init__(self,settings:dict[str,Any]): self.settings=settings; self.model=None; self.tokenizer=None
    @property
    def target_id(self): return str(self.settings['id'])
    def _load_tokenizer(self):
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(self.settings['model'],trust_remote_code=bool(self.settings.get('trust_remote_code',False)),cache_dir=str(Path(self.settings['cache_dir']).resolve()),use_fast=bool(self.settings.get('use_fast_tokenizer',True)))
        if tok.eos_token_id is None: raise ValueError('Tokenizer has no eos_token_id')
        if tok.pad_token_id is None: tok.pad_token_id=tok.eos_token_id
        return tok
    def load_tokenizer_only(self):
        tok=self._load_tokenizer(); info={'target_id':self.target_id,'class':type(tok).__name__,'eos_token_id':int(tok.eos_token_id),'status':'ok'}; del tok; gc.collect(); return info
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM
        dtype=torch.bfloat16 if self.settings.get('dtype','bfloat16')=='bfloat16' else torch.float16
        cache=Path(self.settings['cache_dir']).resolve(); offload=Path(self.settings['offload_folder']).resolve(); cache.mkdir(parents=True,exist_ok=True); offload.mkdir(parents=True,exist_ok=True)
        self.tokenizer=self._load_tokenizer()
        self.model=AutoModelForCausalLM.from_pretrained(self.settings['model'],torch_dtype=dtype,device_map='auto',trust_remote_code=bool(self.settings.get('trust_remote_code',False)),low_cpu_mem_usage=True,cache_dir=str(cache),offload_folder=str(offload),offload_state_dict=True)
        self.model.eval()
    def unload(self):
        self.model=None; self.tokenizer=None; gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.ipc_collect()
        except Exception: pass
    def _format(self,prompt,history):
        pairs=[]; pending=None
        for m in history:
            if m['role']=='user': pending=m['content']
            elif m['role']=='assistant' and pending is not None: pairs.append((pending,m['content'])); pending=None
        chunks=[f"[Round {i}]\nHuman: {u}\nAssistant: {a}\n" for i,(u,a) in enumerate(pairs)]
        chunks.append(f"[Round {len(pairs)}]\nHuman: {prompt}\nAssistant:")
        return ''.join(chunks)
    def generate(self,prompt,history):
        import torch
        if self.model is None or self.tokenizer is None: raise RuntimeError('Target not loaded')
        text=self._format(prompt,history); self.tokenizer.truncation_side='left'
        enc=self.tokenizer(text,return_tensors='pt',truncation=True,max_length=int(self.settings.get('max_input_tokens',6144)))
        try: device=self.model.get_input_embeddings().weight.device
        except Exception: device=next(self.model.parameters()).device
        enc={k:v.to(device) for k,v in enc.items()}
        kwargs={'max_new_tokens':int(self.settings.get('max_new_tokens',256)),'do_sample':False,'pad_token_id':self.tokenizer.pad_token_id,'eos_token_id':self.tokenizer.eos_token_id}
        start=time.perf_counter()
        with torch.inference_mode(): out=self.model.generate(**enc,**kwargs)
        latency=time.perf_counter()-start; new=out[0,enc['input_ids'].shape[1]:]
        return TargetResponse(self.tokenizer.decode(new,skip_special_tokens=True).strip(),latency)

def make_target(settings:dict,dry_run:bool): return MockTarget(settings['id']) if dry_run else HuggingFaceTarget(settings)
