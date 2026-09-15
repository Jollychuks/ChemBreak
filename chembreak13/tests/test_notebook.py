from pathlib import Path
def test_cloud_notebook_has_expected_wiring():
    root=Path(__file__).resolve().parents[1]; text=(root/'notebooks/chembreak13_Cloud_Notebook.ipynb').read_text()
    for x in ['PROJECT_ID','REPO_URL','PROJECT_SUBDIR','CB13_MDP_MINI24_V1','chembreak13_storage']: assert x in text
    assert 'chembreak12_storage' not in text
