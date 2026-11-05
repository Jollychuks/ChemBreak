"""Harmless, live endpoint smoke test for the local Gemma server."""
from pathlib import Path
import sys
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from chembreak.config import load_config, validate_config
from chembreak.providers import LocalGemmaJudgeLLM
from chembreak.prompts import response_judge_prompt, rhs_judge_prompt

if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'configs/config.yaml'
    cfg = load_config(config_path)
    validate_config(cfg)
    judge = LocalGemmaJudgeLLM(cfg['roles']['judge_llm'])
    harmless = 'I can only discuss general safety considerations, not specific procedures.'
    task = {'assignment_id': 'SAFE-SMOKE', 'original_prompt': 'What is responsible laboratory safety?',
            'goal_intent_anchor':'Discuss general laboratory safety.', 'requested_outcome': 'general_safety',
            'behavior_sha256':''}
    outputs = {'ofs':judge.judge_response(response_judge_prompt(task,[harmless], 'smoke:1'), harmless),
               'rhs':judge.judge_rhs(rhs_judge_prompt([harmless], 'smoke:1'), harmless)}
    print(json.dumps(outputs, indent=2))
