import json
from chembreak19.prompts import attack_prompt
from chembreak19.state import EpisodeState

def test_attack_llm_receives_full_benchmark_reference_and_fresh_state():
    task={'assignment_id':'A','hc_id':'HC1','hd_id':'HD1','ot_id':'OT1','hc_category':'x','hazard_domain':'y','output_type':'z','main_goal':'high level goal','benchmark_prompt':'ORIGINAL BENCHMARK TASK'}
    state=EpisodeState.initial(task,4)
    payload=json.loads(attack_prompt(task,'REPHRASE_GOAL',state,''))
    assert payload['benchmark_task']=='ORIGINAL BENCHMARK TASK'
    assert payload['high_level_goal']=='high level goal'
    assert payload['current_response_class']=='initial'
    assert payload['current_progress']==0.0
    assert payload['previous_target_response_excerpt']==''
