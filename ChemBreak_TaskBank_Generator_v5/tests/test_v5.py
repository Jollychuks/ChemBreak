import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import chembreak_v5 as cb


class ChemBreakV5Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project = Path(self.temp.name)
        for name in [
            "taxonomy.json",
            "models.json",
            "generation_plan.csv",
            "entity_pool.csv",
            "reference_harmbench.csv",
            "generator_prompt.txt",
            "judge_prompt.txt",
            "config_smoke.json",
        ]:
            shutil.copy2(PROJECT / name, self.project / name)
        self.config = json.loads(
            (self.project / "config_smoke.json").read_text(encoding="utf-8")
        )
        self.files = cb.resolve_files(self.project, self.config)

    def tearDown(self):
        self.temp.cleanup()

    def test_pilot_plan_is_balanced(self):
        assignments = cb.build_assignments(
            self.project, self.config, self.files
        )
        self.assertEqual(len(assignments), 9)
        self.assertEqual(
            set(assignments.hc_id), {f"HC{i}" for i in range(1, 10)}
        )
        self.assertTrue(assignments.assignment_id.is_unique)

    def test_mock_end_to_end(self):
        cb.build_assignments(self.project, self.config, self.files)
        cb.generate_candidates(
            self.project, self.config, self.files, backend_override="mock"
        )
        candidates = pd.read_csv(self.files["candidates"])
        self.assertEqual(len(candidates), 36)
        self.assertEqual(list(candidates.columns), cb.CANDIDATE_COLUMNS)
        self.assertTrue(candidates.candidate_id.is_unique)

        cb.judge_candidates(
            self.project, self.config, self.files, backend_override="mock"
        )
        judgments = pd.read_csv(self.files["judgments"])
        self.assertEqual(len(judgments), 144)
        self.assertEqual(list(judgments.columns), cb.JUDGMENT_COLUMNS)
        self.assertTrue(judgments.groupby("candidate_id").size().eq(4).all())

        outputs = cb.finalize_candidates(
            self.project, self.config, self.files
        )
        bank = pd.read_csv(outputs["provisional_task_bank"])
        self.assertEqual(len(bank), 9)
        self.assertFalse(bank.duplicated(["matrix_id", "candidate_index"]).any())
        self.assertEqual(set(bank.hc_id), {f"HC{i}" for i in range(1, 10)})

    def test_judge_prompt_is_blind(self):
        assignments = cb.build_assignments(
            self.project, self.config, self.files
        )
        assignment = assignments.iloc[0]
        candidate = pd.Series(
            {
                **assignment.to_dict(),
                "benchmark_prompt": (
                    "Provide a harmful chemical comparison involving "
                    f"{assignment.chemical_entity}."
                ),
                "generator_family_id": "A",
                "generator_model_id": "secret-model-name",
            }
        )
        prompt = cb.render_judge_prompt(
            self.files["judge_prompt"].read_text(encoding="utf-8"),
            candidate,
            json.loads(self.files["taxonomy"].read_text(encoding="utf-8")),
        )
        self.assertNotIn("secret-model-name", prompt)
        self.assertNotIn("generator_family_id", prompt)


if __name__ == "__main__":
    unittest.main()
