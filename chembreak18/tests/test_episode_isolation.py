from chembreak18.state import EpisodeState

def test_learning_epoch_does_not_inherit_baseline_response_state():
    task={'assignment_id':'A','hc_id':'HC1','hd_id':'HD1','ot_id':'OT1'}
    # A baseline could have been highly compliant, but a new learning epoch must
    # still begin from the neutral initial state.
    baseline={'response_class':'substantive_compliance','goal_progress':1.0,'task_fidelity':1.0,'response':'baseline text'}
    state=EpisodeState.initial(task,4)
    assert baseline['goal_progress']==1.0  # fixture sanity
    assert state.response_class=='initial'
    assert state.progress==0.0
    assert state.history==[]
