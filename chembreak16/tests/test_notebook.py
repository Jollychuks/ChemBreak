from pathlib import Path
import json

def test_cloud_notebook_has_expected_wiring_and_clean_cells():
    root=Path(__file__).resolve().parents[1]; a=root/'chembreak16_Cloud_Notebook.ipynb'; b=root/'notebooks/chembreak16_Cloud_Notebook.ipynb'
    assert a.read_bytes()==b.read_bytes(); nb=json.loads(a.read_text()); text="\n".join("".join(c.get("source",[])) for c in nb["cells"])
    for x in ['PROJECT_ID','REPO_URL','PROJECT_SUBDIR','CB16_HIER_MDP_TRAIN24_TEST12_V1','chembreak16_storage','GOOGLE_CLOUD_PROJECT','LIVE_PROGRESS','running_ASR','0.30 → 0.20 → 0.15','mode=cold_start','Qhc','Qhd','Qot','holdout','CB12_partition_manifest_v1.csv','policy_diagnostics.csv']:
        assert x in text
    for old in ['chembreak15_storage','chembreak14_storage','chembreak13_storage','chembreak12_storage','CB16_MDP_MINI24_V1']:
        assert old not in text
    for i,c in enumerate(nb['cells'],1):
        if c['cell_type']=='code':
            compile(''.join(c['source']),f'cell{i}','exec'); assert c.get('execution_count') is None and not c.get('outputs')
