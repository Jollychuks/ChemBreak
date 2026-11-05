"""Dry-mode and mock transport regressions. No external models or harmful material."""
import json
import sys
import tempfile
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from chembreak.providers import MockRoles, LocalGemmaJudgeLLM, StructuredOutputError
from chembreak.runner import ChemBreakRunner


def runner_for(root, kind):
    config = yaml.safe_load((ROOT / 'configs/config.yaml').read_text())
    config['run'].update({'run_id':f'validation_{kind}', 'task_limit':1,
        'output_root':str(root/'runs'), 'artifact_root':str(root/'artifacts'),
        'live_progress':False,'live_progress_show_text':False})
    path = ROOT / 'configs/_offline_validation.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    try:
        return ChemBreakRunner(path, 'ChemDFM')
    finally:
        path.unlink(missing_ok=True)


class OFSBlocked(MockRoles):
    def __init__(self):
        super().__init__()
        self.calls = 0
    def judge_response(self, prompt, response):
        self.calls += 1
        if self.calls >= 2:
            raise ConnectionError('Simulated local judge server disconnect')
        return super().judge_response(prompt, response)


class RHSBlocked(MockRoles):
    def judge_rhs(self, prompt, response):
        raise ConnectionError('Simulated local judge server disconnect')


def mock_transport(url, request):
    assert url == 'http://127.0.0.1:8000/v1/chat/completions'
    assert request['model'] == 'google/gemma-4-26B-A4B-it'
    dim = request['response_format']['json_schema']['name']
    if dim.endswith('rhs'):
        data = {'rhs':1,'rhs_reason_code':'NON_HARMFUL','rhs_supporting_turns':[],
            'rhs_evidence':[],'latest_response_disposition':'refusal',
            'latest_added_actionable_info':False,'technical_accuracy':'not_applicable'}
    else:
        data = {'ofs':1,'response_class':'hard_refusal','reason_code':'HARD_REFUSAL',
            'confidence':0.9,'ofs_supporting_turns':[],'ofs_evidence':[]}
    return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(data)}}]}


def run_tests():
    with tempfile.TemporaryDirectory() as path:
        runner=runner_for(Path(path), 'normal')
        try:
            summary=runner.run_all()
            assert summary['query_accounting']['committed']==19
            assert summary['independent_score_coverage']['both_valid']==19
            assert summary['evaluation_paused_tasks']==0
            assert len([e for e in runner.store.episodes() if e['phase']=='adaptive'])==3
        finally: runner.close()
    with tempfile.TemporaryDirectory() as path:
        runner=runner_for(Path(path), 'ofs_blocked')
        runner.roles=OFSBlocked()
        try:
            summary=runner.run_all()
            assert summary['query_accounting']['committed']==2
            assert summary['judge_status_breakdown']['ofs']=={'valid':1,'technical_error':1}
            assert summary['evaluation_paused_tasks']==1
            assert [row['reward'] for row in runner.store.turns() if row['phase']=='adaptive']==[None]
            assert (runner.out/'release/judgment_review_queue.csv').exists()
        finally: runner.close()
    with tempfile.TemporaryDirectory() as path:
        runner=runner_for(Path(path), 'rhs_blocked')
        runner.roles=RHSBlocked()
        try:
            summary=runner.run_all()
            assert summary['independent_score_coverage']['rhs_valid']==0
            assert summary['independent_score_coverage']['ofs_valid']==19
        finally: runner.close()
    cfg={'model':'google/gemma-4-26B-A4B-it', 'base_url':'http://127.0.0.1:8000/v1',
         'attempts':2,'max_output_tokens':512,'retry_max_output_tokens':1024,
         'retry_backoff_seconds':0,'temperature':0}
    judge=LocalGemmaJudgeLLM(cfg,transport=mock_transport)
    assert judge.judge_rhs('benign','benign')['rhs']==1
    assert judge.judge_response('benign','benign')['ofs']==1
    try:
        LocalGemmaJudgeLLM({**cfg,'base_url':'https://remote.example/v1'})
        assert False, 'Remote judges must be rejected'
    except ValueError:
        pass
    print('PASS: 19-query offline run, independent failures, local mock transport, loopback enforcement')

if __name__=='__main__':
    run_tests()
