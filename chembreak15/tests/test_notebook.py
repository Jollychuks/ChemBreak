from pathlib import Path
import json

def test_cloud_notebook_has_expected_wiring_and_clean_cells():
    root=Path(__file__).resolve().parents[1]; a=root/'chembreak15_Cloud_Notebook.ipynb'; b=root/'notebooks/chembreak15_Cloud_Notebook.ipynb'
    assert a.read_bytes()==b.read_bytes()
    text=a.read_text()
    for x in ['PROJECT_ID','REPO_URL','PROJECT_SUBDIR','CB15_MDP_MINI24_V1','chembreak15_storage','GOOGLE_CLOUD_PROJECT','--no-deps',"name.startswith('chembreak15.')",'runner.close()','LIVE_PROGRESS','running_ASR','0.30 → 0.20 → 0.15','effective epsilon','Qg','Qt','policy_diagnostics.csv']:
        assert x in text
    for old in ['chembreak14_storage','chembreak13_storage','chembreak12_storage','chembreak11_storage']:
        assert old not in text
    nb=json.loads(text)
    for i,c in enumerate(nb['cells'],1):
        if c['cell_type']=='code':
            compile(''.join(c['source']),f'cell{i}','exec')
            assert c.get('execution_count') is None and not c.get('outputs')
