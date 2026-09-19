from chembreak20.checkpoint import Store

def test_delete_partial_episode(tmp_path):
    s=Store(tmp_path/'x.db'); s.start_episode('learning',1,'T'); s.save_turn(phase='learning',epoch=1,assignment_id='T',turn_index=1,action_id='A',selection_mode='',state_key='',route_id='',route_rank=None,q_global=0,q_task=0,combined_q=0,route_bonus=0,candidate_hash='',candidate_source='',candidate_gate={},prompt='',response='',judge={},decision={},reward=0,latency_seconds=0); s.delete_episode_and_turns('learning',1,'T'); assert s.episode_status('learning',1,'T') is None and s.turns()==[]; s.close()
