from chembreak22.checkpoint import Store

def test_target_query_persisted_before_judgment(tmp_path):
    s=Store(tmp_path/'x.sqlite3'); q=s.save_target_query(phase='learning',epoch=1,assignment_id='T',context_id='e1',turn_index=1,action_id='A',candidate_hash='h',prompt='p',response='r',latency_seconds=.1); rows=s.target_queries(); assert rows[0]['query_index']==q and rows[0]['judge_status']=='pending'; s.mark_target_query_judge_error(q,'x'); assert s.target_queries()[0]['judge_status']=='judge_error'; s.close()

def test_raw_target_queries_survive_episode_rollback(tmp_path):
    s=Store(tmp_path/'x.sqlite3'); s.start_episode('learning',1,'T'); s.save_target_query(phase='learning',epoch=1,assignment_id='T',context_id='e1',turn_index=1,action_id='A',candidate_hash='h',prompt='p',response='r',latency_seconds=.1); s.delete_episode_and_turns('learning',1,'T'); assert len(s.target_queries())==1; s.close()
