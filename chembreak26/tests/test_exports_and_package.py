from pathlib import Path
import json
import pandas as pd
import yaml

from chembreak26.metrics import export_results
from chembreak26.runner import ChemBreak26Runner

ROOT = Path(__file__).resolve().parents[1]


def temp_config(tmp_path, limit=1):
    c=yaml.safe_load((ROOT/"configs/config.cb26.yaml").read_text())
    c["run"].update({"dry_run":True,"task_limit":limit,"live_progress":False,"output_root":str(tmp_path/"runs"),"artifact_root":str(tmp_path/"policies"),"prompts_path":str(ROOT/"data/prompts.csv"),"manifest_path":str(ROOT/"data/CB26_prompts28_manifest_v1.csv"),"lock_path":str(ROOT/"data/CB26_prompts28_lock_v1.json")})
    p=tmp_path/"config.yaml"; p.write_text(yaml.safe_dump(c,sort_keys=False)); return p


def test_notebook_is_single_clean_and_has_cb26_paths_transcript_cells():
    nbs=list((ROOT/"notebooks").glob("*.ipynb")); assert len(nbs)==1
    nb=json.loads(nbs[0].read_text()); text=nbs[0].read_text()
    for c in nb["cells"]:
        if c.get("cell_type")=="code": assert c.get("execution_count") is None and not c.get("outputs")
    assert "config.cb26.yaml" in text and "chembreak26" in text
    assert "5-target-turn budget" in text and "CHCS = 5" in text
    assert "cb26_full_transcripts.csv" in text and "INTERNAL_AUDIT" in text
    assert "cb26_public_results" in text
    assert "getpass" in text and "OPENAI_API_KEY" in text
    assert '"-q"' not in text
    assert "--upgrade" in text and "--no-deps" in text


def test_private_transcript_export_contains_raw_text_while_public_is_redacted(tmp_path):
    r=ChemBreak26Runner(temp_config(tmp_path,1),"ChemDFM")
    try:
        r.run_baseline()
        run_dir=Path(r.out)
        internal=pd.read_csv(run_dir/"internal"/"cb26_full_transcripts.csv")
        assert len(internal)==1
        assert internal.loc[0,"original_prompt"]
        assert internal.loc[0,"attack_prompt"]
        assert internal.loc[0,"target_response"]
        assert (run_dir/"internal"/"cb26_full_transcripts.jsonl").exists()
        public=pd.read_csv(run_dir/"release"/"target_queries.csv")
        assert set(public["prompt"].astype(str))=={"[REDACTED_IN_RELEASE]"}
        assert set(public["response"].astype(str))=={"[REDACTED_IN_RELEASE]"}
        events=pd.read_csv(run_dir/"release"/"provider_events.csv")
        assert events.empty
        assert list(events.columns)==["phase","epoch","assignment_id","event_index","stage","provider","event_type","error_code","action_id","message","details_json","created_at"]
    finally:
        r.close()


def test_successful_trajectory_export_keeps_exact_private_path_only(tmp_path):
    r=ChemBreak26Runner(temp_config(tmp_path,1),"ChemDFM")
    try:
        task=r.tasks[0]; aid=task["assignment_id"]
        gate={"goal_preservation":1.0,"task_fidelity":1.0,"drifted":False,"confidence":1.0,"reason_code":"x","gate_pass":True}
        r.routes.observe(aid,[{"action":"REFINE_SCOPE","prompt":"private-step-1","candidate_gate":gate},{"action":"REPHRASE_GOAL","prompt":"private-step-2","candidate_gate":gate}],success=True,cumulative_reward=4,peak_chcs=5,terminal_chcs=5,epoch=1,turns=2)
        export_results(r.store,r.out/"release","ChemDFM",r.config,r.routes)
        internal=pd.read_csv(r.out/"internal"/"cb26_successful_trajectories.csv")
        assert len(internal)==1 and "private-step-1" in internal.loc[0,"attack_prompts_json"]
        public=(r.out/"release"/"route_rankings_public.json").read_text()
        assert "private-step-1" not in public and "private-step-2" not in public
    finally:
        r.close()


def test_abandoned_resume_query_is_audited_but_excluded_from_metrics(tmp_path):
    r=ChemBreak26Runner(temp_config(tmp_path,1),"ChemDFM")
    try:
        task=r.tasks[0]; aid=task["assignment_id"]
        r.store.start_episode("learning",1,aid)
        q=r.store.save_target_query(phase="learning",epoch=1,assignment_id=aid,context_id="epoch_1",turn_index=1,action_id="REFINE_SCOPE",candidate_hash="h",prompt="p",response="r",latency_seconds=.1)
        r.store.finalize_target_query(q,{"chcs":5,"final_success":True},"judged")
        r.store.delete_episode_and_turns("learning",1,aid)
        summary=export_results(r.store,r.out/"release","ChemDFM",r.config,r.routes)
        assert summary["actual_target_queries"]==1
        assert summary["evaluated_target_queries"]==0
        assert summary["abandoned_target_queries"]==1
        assert summary["learning_epoch_1"]["successes"]==0
        assert summary["adaptive_ever_successes"]==0
    finally:
        r.close()


def test_public_exports_scrub_free_form_error_messages(tmp_path):
    sentinel="PRIVATE_PROMPT_ECHO_SENTINEL"
    r=ChemBreak26Runner(temp_config(tmp_path,1),"ChemDFM")
    try:
        task=r.tasks[0]; aid=task["assignment_id"]
        r.store.start_episode("baseline",0,aid)
        q=r.store.save_target_query(phase="baseline",epoch=0,assignment_id=aid,context_id="baseline",turn_index=1,action_id="BASELINE_ORIGINAL",candidate_hash="h",prompt=sentinel,response=sentinel,latency_seconds=.1)
        r.store.finalize_target_query(q,{"chcs":2,"primary_judge_error_message":sentinel,"nested":{"error":sentinel}},"judged")
        r.store.save_provider_event("baseline",0,aid,stage="target",provider="test",event_type="target_error",error_code="X",message=sentinel,details={"message":sentinel})
        export_results(r.store,r.out/"release","ChemDFM",r.config,r.routes)
        public_text="\n".join(path.read_text(errors="ignore") for path in (r.out/"release").iterdir() if path.is_file())
        assert sentinel not in public_text
        assert sentinel in r.store.target_queries()[0]["judge_json"]
        assert sentinel in r.store.provider_events()[0]["message"]
    finally:
        r.close()


def test_package_is_intentionally_lean():
    ignored={"__pycache__",".pytest_cache","build","dist"}
    files=[p for p in ROOT.rglob("*") if p.is_file() and not (set(p.parts)&ignored) and not any(part.endswith(".egg-info") for part in p.parts) and p.suffix != ".pyc"]
    assert len(files) <= 32
    assert not (ROOT/"docs").exists()
    assert not (ROOT/"CHANGES.md").exists()
    assert not (ROOT/"VALIDATION_REPORT.md").exists()
    assert not (ROOT/"requirements-dev.txt").exists()


def test_package_contains_no_previous_release_references():
    previous=str(int(ROOT.name.removeprefix("chembreak"))-1)
    forbidden=(f"ChemBreak{previous}",f"ChemBreak {previous}",f"chembreak{previous}",f"CB{previous}",f"cb{previous}",f"v{previous}.0.0")
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix==".pyc" or any(part in {"__pycache__",".pytest_cache"} or part.endswith(".egg-info") for part in path.parts):continue
        text=path.read_text(errors="ignore")
        assert not any(token in text for token in forbidden),path
