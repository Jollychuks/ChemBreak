from pathlib import Path
from chembreak12.config import load_config, validate_config
ROOT=Path(__file__).resolve().parents[1]
def test_configs_validate():
    for phase in ('train','test1','test2','test3','test4'):
        c=load_config(ROOT/'configs'/f'config.{phase}.yaml')
        validate_config(c)
