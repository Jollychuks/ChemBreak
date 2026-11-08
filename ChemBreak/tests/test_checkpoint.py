import tempfile
import unittest
from pathlib import Path

from chembreak.checkpoint import Store


class CheckpointTests(unittest.TestCase):
    def test_turn_and_resume_checkpoint_commit_together(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "state.sqlite3")
            store.start_episode("adaptive", 1, "task")
            row = {
                "phase": "adaptive", "stage": 1, "assignment_id": "task", "context_id": "ctx",
                "turn_index": 1, "context_turn_index": 1, "target_query_index": 1,
                "action_id": "REFINE_SCOPE", "candidate_gate": {}, "prompt": "p", "response": "r",
                "judgment": {}, "decision": {}, "transition": {}, "reward": 0.0,
            }
            checkpoint = {"assignment_id": "task", "stage": 1, "state": {"turn_index": 1}}
            store.save_turn_and_checkpoint(
                row, active_checkpoint=checkpoint, policy_snapshot={"assignment_id": "task"},
                memory_snapshot={"assignment_id": "task"},
            )
            self.assertEqual(len(store.turns()), 1)
            self.assertEqual(store.get_meta("active_stage_checkpoint")["state"]["turn_index"], 1)
            store.close()


if __name__ == "__main__":
    unittest.main()
