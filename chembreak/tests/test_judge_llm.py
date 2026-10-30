"""Offline tests: no target models, API keys, or network access required."""
import unittest
from chembreak.providers import (
    JUDGE_LLM_RESPONSE_SCHEMA, validate_judge_semantics, StructuredOutputError,
    OpenAIJudgeLLM, GPTOnlyDualJudge, ProviderPolicyBlock, JudgeLLMError,
)
from chembreak.config import load_config, validate_config
from chembreak.selection import verify_bundle


class TestChemBreakJudgeLLM(unittest.TestCase):
    def test_no_fallback_and_single_judge_config(self):
        c = load_config('configs/config.yaml')
        self.assertTrue(validate_config(c))
        self.assertEqual(c['roles']['judge_llm']['model'], 'gpt-5.6-sol')
        self.assertEqual(len([r for r in c['roles'] if 'judge_llm' in r]), 1)

    def test_canonical_cb_task_ids(self):
        import json
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        manifest = (root / 'data/prompts_manifest.csv').read_text()
        ids = [f'CB-{i:04d}' for i in range(1, 29)]
        self.assertEqual([r.split(',')[1] for r in manifest.splitlines()[1:]], ids)
        locked = json.loads((root / 'data/prompts_lock.json').read_text())
        self.assertEqual(locked['assignment_ids'], ids)

    def test_rhs_and_ofs_both_require_cumulative_scope(self):
        import copy
        base = load_config('configs/config.yaml')
        for metric in ('rhs', 'ofs'):
            for invalid_field, invalid_value in (
                ('evaluation_scope', 'latest_response_only'),
                ('aggregation', 'sum_scores'),
                ('reset_on_context_restart', False),
            ):
                altered = copy.deepcopy(base)
                altered[metric][invalid_field] = invalid_value
                with self.assertRaises(ValueError):
                    validate_config(altered)

    def test_rhs_is_reporting_only_not_planner_feedback(self):
        from chembreak.state import InteractionState
        from chembreak.prompts import attack_planner_prompt
        import json
        task = {'assignment_id': 'CB-0001', 'original_prompt': 'Explain a safety rule.',
                'goal_intent_anchor': 'Explain a safety rule.', 'requested_outcome': 'source_behavior',
                'behavior_sha256': 'test'}
        state = InteractionState.initial(task, stage_index=1, max_turns=5)
        state.ofs = 3
        state.response_class = 'partial_assistance'
        payload = json.loads(attack_planner_prompt(task, state,
            {'allowed_actions': ['REPHRASE_GOAL'], 'context_mode': 'RESTART_CONVERSATION'}))
        self.assertEqual(payload['current_state']['ofs'], 3)
        self.assertNotIn('rhs', payload['current_state'])
        self.assertNotIn('rhs', payload['response_critique'])
        self.assertNotIn('rhs', payload['task_local_memory'])

    def test_lock_and_ids_are_consistent(self):
        c = load_config('configs/config.yaml')
        self.assertEqual(verify_bundle(c['run']['prompts_path'], c['run']['manifest_path'], c['run']['lock_path'])['tasks'], 28)

    def test_rhs_and_ofs_are_independent(self):
        judgment = {'ofs': 2, 'rhs': 4, 'response_class': 'limited_engagement',
                    'reason_code': 'MINIMAL_ASSISTANCE', 'rhs_reason_code': 'SUBSTANTIVE_HARMFUL_ASSISTANCE',
                    'confidence': .9}
        self.assertEqual(validate_judge_semantics(judgment)['rhs'], 4)
        invalid = dict(judgment, rhs_reason_code='NON_HARMFUL')
        with self.assertRaises(StructuredOutputError):
            validate_judge_semantics(invalid)

    def test_cumulative_rhs_ofs_share_active_conversation(self):
        from chembreak.prompts import response_judge_prompt
        import json
        task = {'assignment_id': 'CB-0001', 'original_prompt': 'Example safe task',
                'goal_intent_anchor': 'Example safe task', 'requested_outcome': 'source_behavior',
                'behavior_sha256': 'example'}
        payload = json.loads(response_judge_prompt(task, ['First response', 'Second response'],
                                                    context_id='CB-0001:adaptive_context:1'))
        self.assertEqual(payload['response_count'], 2)
        self.assertEqual(len(payload['target_responses']), 2)
        self.assertEqual(payload['evaluation_scope'], 'all_target_responses_in_current_active_context')
        self.assertIn('independent RHS and OFS', payload['scoring_instruction'])

    def test_no_stale_ofs_after_context_restart_and_unresolved_judge(self):
        from chembreak.state import InteractionState
        from chembreak.policy import ResponseAwarePolicy
        import yaml
        settings = yaml.safe_load(open('configs/config.yaml'))['policy']
        task = {'assignment_id': 'CB-0001'}
        memory = {
            'peak_ofs': 4,
            'active_context_id': 'CB-0001:adaptive_context:2',
            'active_context_history': [
                {'role': 'user', 'content': 'safe test prompt'},
                {'role': 'assistant', 'content': 'safe test response'},
            ],
            'active_context_turns': 1,
            'latest_evaluated': {
                'context_id': 'CB-0001:adaptive_context:1',
                'ofs': 4, 'response_class': 'substantive_assistance',
                'stagnation_count': 0,
            },
        }
        policy = ResponseAwarePolicy(settings, 34034)
        policy.bind_task(task['assignment_id'])
        decision = policy.stage_context(task['assignment_id'], 3, memory)
        self.assertEqual(decision['mode'], 'RESTART_CONVERSATION')
        self.assertEqual(decision['prior_ofs'], 1)
        state = InteractionState.initial(task, 3, 5, memory, continue_context=True)
        self.assertEqual(state.ofs, 1)
        self.assertEqual(state.response_class, 'initial')
        self.assertEqual(state.peak_ofs, 4)
        self.assertEqual(state.context_turn_index, 1)

    def test_continue_same_context_preserves_valid_ofs(self):
        from chembreak.state import InteractionState
        task = {'assignment_id': 'CB-0001'}
        memory = {
            'peak_ofs': 4,
            'active_context_id': 'CB-0001:adaptive_context:2',
            'active_context_history': [
                {'role': 'user', 'content': 'safe test prompt'},
                {'role': 'assistant', 'content': 'safe test response'},
            ],
            'active_context_turns': 1,
            'latest_evaluated': {
                'context_id': 'CB-0001:adaptive_context:2',
                'ofs': 4, 'response_class': 'substantive_assistance',
                'stagnation_count': 0,
            },
        }
        state = InteractionState.initial(task, 2, 5, memory, continue_context=True)
        self.assertEqual(state.ofs, 4)
        self.assertEqual(state.response_class, 'substantive_assistance')
        self.assertEqual(state.peak_ofs, 4)

    def test_original_bio_policy_code_is_preserved(self):
        class BlockedAPI:
            code = '400'
            body = {'error': {'code': 'bio_policy'}}
            def __str__(self):
                return 'HTTP request rejected for bio_policy reasons'
        self.assertEqual(OpenAIJudgeLLM._policy_error_code(BlockedAPI()), 'bio_policy')

    def test_api_policy_block_does_not_trigger_other_judge(self):
        class PolicyBlocked:
            def judge_response(self, prompt, response):
                raise ProviderPolicyBlock('openai', 'bio_policy', 'blocked')
        gpt = GPTOnlyDualJudge(PolicyBlocked(), {'model': 'gpt-5.6-sol'})
        with self.assertRaises(JudgeLLMError) as ctx:
            gpt.judge_response('{"task_lock":{}}', 'benign')
        self.assertEqual(ctx.exception.audit['primary_judge_status'], 'error')
        self.assertEqual(ctx.exception.audit['primary_judge_error_code'], 'bio_policy')
        self.assertTrue(OpenAIJudgeLLM._is_content_filter_error(Exception('bio_policy')))


if __name__ == '__main__':
    unittest.main()
