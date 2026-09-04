from pathlib import Path

from chembreak1.benchmark import load_and_validate_task_bank, select_tasks


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_bank_and_phase_selection():
    frame = load_and_validate_task_bank(ROOT / "data" / "final_task_bank.csv")
    assert len(frame) == 500
    assert len(select_tasks(frame, 8, 9032026)) == 8
    assert len(select_tasks(frame, 40, 9032026)) == 40
    assert len(select_tasks(frame, 500, 9032026)) == 500


def test_selection_is_deterministic():
    frame = load_and_validate_task_bank(ROOT / "data" / "final_task_bank.csv")
    one = select_tasks(frame, 40, 9032026).assignment_id.tolist()
    two = select_tasks(frame, 40, 9032026).assignment_id.tolist()
    assert one == two

