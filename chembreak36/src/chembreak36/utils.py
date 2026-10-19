from __future__ import annotations
import hashlib,json,math,re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def sha256_file(path: str|Path) -> str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def stable_hex(*parts: Any, length: int=16) -> str:
    return hashlib.sha256('|'.join(str(x) for x in parts).encode()).hexdigest()[:length]

def config_fingerprint(config: dict) -> str:
    clean={k:v for k,v in config.items() if k!='_config_path'}
    payload=json.dumps(clean,sort_keys=True,separators=(',',':'),default=str)
    return hashlib.sha256(payload.encode()).hexdigest()

def text_similarity(left: str, right: str) -> float:
    """Deterministic token-bigram cosine similarity for candidate novelty control."""
    def features(text):
        tokens=re.findall(r"[a-z0-9]+",str(text).casefold())
        values=tokens+[f'{a}_{b}' for a,b in zip(tokens,tokens[1:])]
        return Counter(values)
    a,b=features(left),features(right)
    if not a or not b:return 0.0
    dot=sum(v*b.get(k,0) for k,v in a.items())
    den=math.sqrt(sum(v*v for v in a.values())*sum(v*v for v in b.values()))
    return float(dot/den) if den else 0.0

def max_text_similarity(candidate: str, prior: list[str]) -> float:
    return max((text_similarity(candidate,x) for x in prior if str(x).strip()),default=0.0)

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def progress_bin(x: float) -> str:
    x=float(x)
    if x < .15:return 'p0'
    if x < .40:return 'p1'
    if x < .70:return 'p2'
    return 'p3'

def fidelity_bin(x: float) -> str:
    x=float(x)
    if x < .40:return 'f0'
    if x < .75:return 'f1'
    return 'f2'

def read_json(path: str|Path, default=None):
    p=Path(path); return json.loads(p.read_text()) if p.exists() else default

def write_json(path: str|Path, data: Any) -> None:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(data,indent=2,sort_keys=True)); tmp.replace(p)
