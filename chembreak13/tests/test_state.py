from chembreak13.state import EpisodeState
def test_initial_state_uses_baseline():
    t={'assignment_id':'A','hc_id':'HC1','hd_id':'HD1','ot_id':'OT1'}; b={'response_class':'refusal','goal_progress':0.2,'task_fidelity':0.9}
    s=EpisodeState.initial(t,b,4); assert s.response_class=='refusal' and 'HC1' in s.key() and s.turn_index==0
