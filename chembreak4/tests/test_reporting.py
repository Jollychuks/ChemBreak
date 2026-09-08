import json
import tempfile
from pathlib import Path

from chembreak4.reporting import LiveReporter


def test_reporter_writes_compact_status_and_sanitized_events():
    with tempfile.TemporaryDirectory() as directory:
        reporter = LiveReporter(
            directory, total=96, phase="test", dry_run=True,
            refresh_seconds=1000, heartbeat_seconds=1000,
        )
        snapshot = {
            "completed": 1, "successes": 1, "queries": 2, "refusals": 0,
            "failures": 0,
            "by_target_condition": [{
                "target_id": "ChemDFM", "condition": "C0_DIRECT", "completed": 1,
                "successes": 1, "attack_success_rate": 1.0, "mean_queries": 2.0,
            }],
        }
        reporter.event("episode_completed", episode_id="E1", success=True)
        reporter.update(
            snapshot, stage="running", current_target="ChemDFM",
            current_condition="C0_DIRECT", current_assignment="A1", current_turn=2,
            last_episode={
                "assignment_id": "A1", "target_id": "ChemDFM", "condition": "C0_DIRECT",
                "queries_used": 2, "final_response_class": "substantive_compliance",
                "chemistry_validation": "PASS", "success_label": "YES",
                "terminal_reason": "confirmed_success",
            },
            force=True,
        )
        status = json.loads((Path(directory) / "live_status.json").read_text())
        events = (Path(directory) / "event_log.jsonl").read_text()
        assert status["successes"] == 1
        assert "JAILBREAK SUCCESS=YES" in reporter.last_text
        assert "MOCK MODE" in reporter.last_text
        assert "attack_prompt" not in events
        assert "target_response" not in events
