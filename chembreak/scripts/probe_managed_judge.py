"""Benign smoke test for the managed Gemma endpoint, using Colab ADC."""
import os, sys
from pathlib import Path
from chembreak.config import load_config, validate_config
from chembreak.providers import ManagedGemmaJudgeLLM

config_path = sys.argv[1] if len(sys.argv)>1 else 'configs/config.yaml'
cfg = load_config(Path(config_path))
validate_config(cfg)
project_id = os.environ.get('GOOGLE_CLOUD_PROJECT','').strip()
if not project_id:
    raise RuntimeError('Set GOOGLE_CLOUD_PROJECT before probing the managed model')
print(ManagedGemmaJudgeLLM(cfg['roles']['judge_llm'], project_id).probe_connection())
