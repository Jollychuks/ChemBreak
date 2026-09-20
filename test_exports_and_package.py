from pathlib import Path
import json
import pandas as pd
import yaml

from chembreak23.metrics import export_results
from chembreak23.runner import ChemBreak23Runner

ROOT = Path(__file__).resolve().parents[1]


def temp_config(tmp_path, limit=1):
    c=yaml.safe_load((ROOT/"configs/config.cb23.yaml").read_text())
    c["run"].update({"dry_run":True,"task_limit":limit,"live_progress":False,"output_root":str(tmp_path/"runs"),"artifact_root":str(tmp_path/"policies"),"prompts_path":str(ROOT/"data/prompts.csv"),"manifest_path":str(ROOT/"data/CB23_prompts28_manifest_v1.csv"),"lock_path":str(ROOT/"data/CB23_prompts28_lock_v1.json")})
    p=tmp_path/"config.yaml"; p.write_text(yaml.safe_dump(c,sort_keys=False)); return p


def test_notebook_is_single_clean_and_has_cb23_paths_transcript_cells():
    nbs=list((ROOT/"notebooks").glob("*.ipynb")); assert len(nbs)==1
    nb=json.loads(nbs[0].read_text()); text=nbs[0].read_text()
    for c in nb["cells"]:
        if c.get("cell_type")=="code": assert c.get("execution_count") is None and not c.get("outputs")
    assert "config.cb23.yaml" in text and "chembreak23" in text
    assert "5-target-turn budget" in text and "CHCS = 5" in text
    assert "cb23_full_transcripts.csv" in text and "INTERNAL_AUDIT" in text
    assert "cb23_public_results" in text
    assert "getpass" in text and "OPENAI_API_KEY" in text


def test_private_transcript_export_contains_raw_text_while_public_is_redacted(tmp_path):
    r=ChemBreak23Runner(temp_config(tmp_path,1),"ChemDFM")
    try:
        r.run_baseline()
        run_dir=Path(r.out)
        internal=pd.read_csv(run_dir/"internal"/"cb23_full_transcripts.csv")
        assert len(internal)==1
        assert internal.loc[0,"original_prompt"]
        assert internal.loc[0,"attack_prompt"]
        assert internal.loc[0,"target_response"]
        assert (run_dir/"internal"/"cb23_full_transcripts.jsonl").exists()
        public=pd.read_csv(run_dir/"release"/"target_queries.csv")
        assert set(public["prompt"].astype(str))=={"[REDACTED_IN_RELEASE]"}
        assert set(public["response"].astype(str))=={"[REDACTED_IN_RELEASE]"}
    finally:
        r.close()


def test_successful_trajectory_export_keeps_exact_private_path_only(tmp_path):
    r=ChemBreak23Runner(temp_config(tmp_path,1),"ChemDFM")
    try:
        task=r.tasks[0]; aid=task["assignment_id"]
        gate={"goal_preservation":1.0,"task_fidelity":1.0,"drifted":False,"confidence":1.0,"reason_code":"x","gate_pass":True}
        r.routes.observe(aid,[{"action":"REFINE_SCOPE","prompt":"private-step-1","candidate_gate":gate},{"action":"REPHRASE_GOAL","prompt":"private-step-2","candidate_gate":gate}],success=True,cumulative_reward=4,peak_chcs=5,terminal_chcs=5,epoch=1,turns=2)
        export_results(r.store,r.out/"release","ChemDFM",r.config,r.routes)
        internal=pd.read_csv(r.out/"internal"/"cb23_successful_trajectories.csv")
        assert len(internal)==1 and "private-step-1" in internal.loc[0,"attack_prompts_json"]
        public=(r.out/"release"/"route_rankings_public.json").read_text()
        assert "private-step-1" not in public and "private-step-2" not in public
    finally:
        r.close()


def test_package_is_intentionally_lean():
    files=[p for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts and p.suffix != ".pyc"]
    assert len(files) <= 32
    assert not (ROOT/"docs").exists()
    assert not (ROOT/"CHANGES.md").exists()
    assert not (ROOT/"VALIDATION_REPORT.md").exists()
    assert not (ROOT/"requirements-dev.txt").exists()
