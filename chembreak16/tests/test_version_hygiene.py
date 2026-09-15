from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _prior_tokens():
    tokens=[]
    for i in range(1,16):
        tokens.extend([
            f'chembreak{i}_', f'chembreak{i}.', f'chembreak{i}-', f'chembreak{i}/',
            f'cb{i}_', f'cb{i}.', f'cb{i}-', f'cb{i}/',
            f'/content/chembreak{i}_storage',
        ])
    return tuple(tokens)


def test_cb16_has_no_prior_runtime_names_or_artifact_files():
    forbidden=_prior_tokens()
    for path in ROOT.rglob('*'):
        if path.is_file():
            low=path.name.lower()
            assert not any(tok in low for tok in forbidden), path

    # Source task identifiers from the frozen bank are immutable data identities
    # and are intentionally excluded from runtime-name checks.
    checked_roots=[ROOT/'src',ROOT/'scripts',ROOT/'configs',ROOT/'docs',ROOT/'notebooks']
    for base in checked_roots:
        for path in base.rglob('*'):
            if not path.is_file():
                continue
            text=path.read_text(errors='ignore').lower()
            assert not any(tok in text for tok in forbidden), path
