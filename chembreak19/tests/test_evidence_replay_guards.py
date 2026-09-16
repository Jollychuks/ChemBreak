from .test_dry_run import config
from chembreak19.runner import ChemBreak19Runner


def test_cb19_candidate_evidence_is_diagnostic_not_replayed(tmp_path):
    r=ChemBreak19Runner(config(tmp_path,1))
    try:
        task=r.tasks[0]; aid=str(task['assignment_id'])
        r.evidence.record(task_id=aid,prompt='candidate only',action_id='REFINE_SCOPE',global_key='G',task_key='K',phase='learning',epoch=1,turn_index=1,success=True,reward=2.0,combined_q=1.0,source='attack_llm_fresh')
        assert r.config['evidence']['replay_during_learning'] is False
        assert r.trajectories.best_live_trajectory(aid) is None
    finally:
        r.close()
