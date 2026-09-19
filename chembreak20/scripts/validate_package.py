from __future__ import annotations
import ast,json,re,sys
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]


def main():
    errors=[]
    # Check package hygiene before importing the package, because Python imports themselves create __pycache__.
    initial_caches=[p for p in ROOT.rglob('*') if p.name in {'__pycache__','.pytest_cache'} or p.suffix=='.pyc']
    if initial_caches:errors.append(f'cache artifacts present before validation: {len(initial_caches)}')
    sys.path.insert(0,str(ROOT/'src'))
    from chembreak20.config import load_config,validate_config
    from chembreak20.selection import verify_bundle
    cfg=load_config(ROOT/'configs/config.cb20.yaml')
    try:validate_config(cfg)
    except Exception as exc:errors.append(f'config: {exc}')
    try:sel=verify_bundle(cfg['run']['task_bank_path'],cfg['run']['mini_manifest_path'],cfg['run']['mini_lock_path'])
    except Exception as exc:errors.append(f'selection: {exc}'); sel=None

    notebooks=list((ROOT/'notebooks').glob('*.ipynb'))
    if len(notebooks)!=1:errors.append(f'expected exactly one notebook, found {len(notebooks)}')
    nb_cells=0
    if len(notebooks)==1:
        nb=json.loads(notebooks[0].read_text()); nb_cells=len(nb.get('cells',[]))
        for i,c in enumerate(nb.get('cells',[]),1):
            if c.get('cell_type')=='code':
                if c.get('outputs'):errors.append(f'notebook cell {i} contains saved output')
                if c.get('execution_count') is not None:errors.append(f'notebook cell {i} has execution_count')
                try:ast.parse(''.join(c.get('source',[])))
                except SyntaxError as exc:errors.append(f'notebook cell {i} syntax: {exc}')
    for p in ROOT.rglob('*.py'):
        try:ast.parse(p.read_text())
        except SyntaxError as exc:errors.append(f'python syntax {p.relative_to(ROOT)}: {exc}')
    secrets=[]
    pat=re.compile(r'sk-[A-Za-z0-9_-]{16,}')
    for p in ROOT.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.py','.yaml','.yml','.json','.md','.toml','.txt','.ipynb'}:
            txt=p.read_text(errors='ignore')
            if pat.search(txt):secrets.append(str(p.relative_to(ROOT)))
    if secrets:errors.append(f'possible embedded OpenAI key in {secrets}')
    old=[]
    old_pats=[re.compile(r'chembreak(?:1[0-9])_(?:storage|repo)',re.I),re.compile(r'config\.cb1[0-9]',re.I),re.compile(r'CB1[0-9]_(?:EVIDENCE|TRAJECTORY|HIER|MDP)',re.I)]
    for p in ROOT.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.py','.yaml','.yml','.json','.md','.toml','.ipynb'}:
            txt=p.read_text(errors='ignore')
            for rx in old_pats:
                if rx.search(txt):old.append(f'{p.relative_to(ROOT)}:{rx.pattern}')
    if old:errors.append(f'old runtime namespace references: {old[:10]}')
    targets={x['id']:x for x in cfg['targets']}
    if targets.get('ChemDFM',{}).get('template')!='round_chat':errors.append('ChemDFM template mismatch')
    if targets.get('ChemLLM',{}).get('template')!='tokenizer_chat_template':errors.append('ChemLLM template mismatch')
    report={'status':'ok' if not errors else 'failed','errors':errors,'selection':sel,'notebooks':len(notebooks),'notebook_cells':nb_cells,'targets':{k:{'model':v['model'],'template':v['template']} for k,v in targets.items()},'attack_llm':cfg['roles']['attack_llm']['model'],'judge_llm':cfg['roles']['judge_llm']['model']}
    print(json.dumps(report,indent=2,sort_keys=True))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
