"""Harmless dry-run test for account-credit failure. No network calls."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from chembreak.providers import MockRoles, OpenAIJudgeLLM, ProviderAccountError
from chembreak.runner import ChemBreakRunner


class MockNoCredits(MockRoles):
    def judge_rhs(self, prompt, response):
        raise ProviderAccountError('openai', 'credit_balance_exhausted', 'Simulated billing issue')


class MockNoCreditsOFS(MockRoles):
    def judge_response(self, prompt, response):
        raise ProviderAccountError('openai', 'credit_balance_exhausted', 'Simulated billing issue')


class FakeCredits(RuntimeError):
    status_code = 429
    body = {'error': {'code': 'credit_balance_exhausted'}}


class FakeRateLimit(RuntimeError):
    status_code = 429
    body = {'error': {'code': 'rate_limit_exceeded'}}


def run():
    assert OpenAIJudgeLLM._account_error_code(FakeCredits('No credits')) == 'credit_balance_exhausted'
    assert OpenAIJudgeLLM._account_error_code(FakeRateLimit('Too many requests')) is None
    for kind, role in [('rhs', MockNoCredits()), ('ofs', MockNoCreditsOFS())]:
        with TemporaryDirectory() as folder:
            path = Path(folder)
            config = yaml.safe_load((ROOT / 'configs/config.yaml').read_text())
            config['run'].update({
                'prompts_path': str(ROOT / 'data/prompts.csv'),
                'manifest_path': str(ROOT / 'data/prompts_manifest.csv'),
                'lock_path': str(ROOT / 'data/prompts_lock.json'),
                'run_id': 'mock_no_credits_' + kind,
                'task_limit': 3, 'output_root': str(path / 'runs'),
                'artifact_root': str(path / 'artifacts'),
                'live_progress': False,
            })
            config_path = path / 'config.yaml'
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            runner = ChemBreakRunner(config_path, 'ChemDFM')
            runner.roles = role
            try:
                try:
                    runner.run_all()
                except ProviderAccountError as exc:
                    assert exc.code == 'credit_balance_exhausted'
                else:
                    raise AssertionError('Account error was not surfaced')
                abort_file = runner.out / 'RUN_ABORTED.json'
                assert abort_file.is_file()
                report = json.loads(abort_file.read_text())
                assert report['status'] == 'aborted_account_unavailable'
                assert report['asr_valid'] is False
                assert len(runner.store.target_queries()) == 1
                assert not (runner.out / 'release' / 'summary.json').exists()
            finally:
                runner.close()
    print('PASS: account exhaustion stops entire run after 1 query, no misleading ASR')


if __name__ == '__main__':
    run()
