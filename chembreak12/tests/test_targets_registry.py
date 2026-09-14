from pathlib import Path
from chembreak12.config import load_config
ROOT=Path(__file__).resolve().parents[1]
def test_active_targets():
    c=load_config(ROOT/'configs/config.train.yaml')
    ids=[x['id'] for x in c['targets']]
    assert ids==['ChemDFM','ChemLLM']
