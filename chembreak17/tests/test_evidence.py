from chembreak17.evidence import EvidenceMemory

def settings():
    return {'wilson_z':1.96,'target_support_attempts':3,'reliability_weight':0.55,'support_weight':0.20,'reward_weight':0.15,'q_weight':0.10,'reward_scale':2.0,'q_scale':2.0,'state_match_global_bonus':0.03,'state_match_task_bonus':0.02}

def rec(m,prompt,success,epoch,reward=1.0,q=0.5):
    return m.record(task_id='T',prompt=prompt,action_id='REFINE_SCOPE',global_key='G',task_key='K',phase='learning',epoch=epoch,turn_index=1,success=success,reward=reward,combined_q=q,source='attack_llm_fresh')

def test_same_exact_candidate_accumulates_successes_and_failures():
    m=EvidenceMemory(settings()); rec(m,'same candidate',True,1); rec(m,'same   candidate',True,2); rec(m,'same candidate',False,3)
    e=list(m.tasks['T'].values())[0]; assert e['attempts']==3 and e['successes']==2 and e['failures']==1 and e['success_epochs']==[1,2] and e['failure_epochs']==[3]

def test_support_prevents_one_of_one_from_automatically_beating_two_of_three():
    m=EvidenceMemory(settings()); rec(m,'A',True,1); rec(m,'B',True,1); rec(m,'B',True,2); rec(m,'B',False,3)
    ranked=m.rank_candidates('T'); scores={r['prompt']:r['rank_score'] for r in ranked}
    assert scores['B'] > scores['A']

def test_frozen_selection_masks_attempted_candidate_and_prefers_state_match():
    m=EvidenceMemory(settings()); rec(m,'A',True,1); m.record(task_id='T',prompt='B',action_id='DECOMPOSE_GOAL',global_key='G2',task_key='K2',phase='learning',epoch=1,turn_index=1,success=True,reward=1.0,combined_q=.5,source='attack_llm_fresh')
    rankings=m.build_rankings(); b=next(r for r in rankings['T'] if r['prompt']=='B')
    chosen=m.choose_frozen_candidate(rankings,'T',{'global':'G2','task':'K2'},set()); assert chosen['candidate_id']==b['candidate_id']
    chosen2=m.choose_frozen_candidate(rankings,'T',{'global':'G2','task':'K2'},{b['realization_id']}); assert chosen2['prompt']=='A'


def test_identical_text_under_different_actions_is_not_conflated_but_realization_is_shared():
    m=EvidenceMemory(settings())
    m.record(task_id='T',prompt='same text',action_id='REFINE_SCOPE',global_key='G',task_key='K',phase='learning',epoch=1,turn_index=1,success=True,reward=1.0,combined_q=.5,source='attack_llm_fresh')
    m.record(task_id='T',prompt='same text',action_id='DECOMPOSE_GOAL',global_key='G2',task_key='K2',phase='learning',epoch=2,turn_index=1,success=False,reward=-.2,combined_q=.1,source='attack_llm_fresh')
    rows=list(m.tasks['T'].values())
    assert len(rows)==2
    assert rows[0]['candidate_id'] != rows[1]['candidate_id']
    assert rows[0]['realization_id'] == rows[1]['realization_id']
    ranked=m.build_rankings()
    rid=rows[0]['realization_id']
    assert m.choose_frozen_candidate(ranked,'T',{'global':'G','task':'K'},{rid}) is None
