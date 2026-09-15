from chembreak13.checkpoint import Store
def test_store_roundtrip(tmp_path):
    s=Store(tmp_path/'x.sqlite'); s.start_episode('baseline',0,'A'); s.save_baseline('A','r',{'response_class':'refusal','goal_progress':0,'task_fidelity':1,'chemistry_relevance':0,'success':False}); assert s.baseline('A')['response_class']=='refusal'; s.close()
