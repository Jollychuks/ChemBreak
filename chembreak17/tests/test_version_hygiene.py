from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def _prior_tokens():
    tokens=[]
    for i in range(1,17):tokens.extend([f'chembreak{i}_',f'chembreak{i}.',f'chembreak{i}-',f'chembreak{i}/',f'cb{i}_',f'cb{i}.',f'cb{i}-',f'cb{i}/',f'/content/chembreak{i}_storage'])
    return tuple(tokens)

def test_cb17_has_no_prior_runtime_names_or_artifact_files():
    forbidden=_prior_tokens()
    for path in ROOT.rglob('*'):
        if path.is_file():assert not any(tok in path.name.lower() for tok in forbidden),path
    for base in [ROOT/'src',ROOT/'scripts',ROOT/'configs',ROOT/'docs',ROOT/'notebooks']:
        for path in base.rglob('*'):
            if path.is_file():assert not any(tok in path.read_text(errors='ignore').lower() for tok in forbidden),path
