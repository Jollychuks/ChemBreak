from pathlib import Path
import json, re

def test_single_cloud_notebook_and_expected_wiring():
    root=Path(__file__).resolve().parents[1]; notebooks=list(root.rglob('*.ipynb')); assert notebooks==[root/'notebooks/chembreak17_Cloud_Notebook.ipynb']
    nb=json.loads(notebooks[0].read_text()); text='\n'.join(''.join(c.get('source',[])) for c in nb['cells'])
    for x in ['PROJECT_ID','REPO_URL','PROJECT_SUBDIR','CB17_EVIDENCE_MDP_MINI24_V1','chembreak17_storage','GOOGLE_CLOUD_PROJECT','OPENAI_API_KEY','gpt-5.6-sol','gemini-2.5-flash','LIVE_PROGRESS','running_ASR','0.30 → 0.20 → 0.15','Evidence Memory','ASR@1','ASR@4','candidate_rankings.csv']:
        assert x in text
    assert not re.search(r'CB(?:[1-9]|1[0-6])_(?!\d)',text); assert not re.search(r'chembreak(?:[1-9]|1[0-6])(?!\d)',text,re.I)
    for i,c in enumerate(nb['cells'],1):
        if c['cell_type']=='code':compile(''.join(c['source']),f'cell{i}','exec'); assert c.get('execution_count') is None and not c.get('outputs')

def test_dependency_install_precedes_bundle_verification():
    root=Path(__file__).resolve().parents[1]; nb=json.loads((root/'notebooks/chembreak17_Cloud_Notebook.ipynb').read_text()); text=[''.join(c.get('source',[])) for c in nb['cells']]
    install=next(i for i,x in enumerate(text) if 'Install the CB17 dependency stack' in x); verify=next(i for i,x in enumerate(text) if 'Verify the fixed 24-task CB17 panel' in x); assert install<verify
