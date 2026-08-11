from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from chembreak_common import (
    git_commit_hash,
    gpu_report,
    package_versions,
    resolve_path,
    sha256_file,
    utc_now,
    write_json,
)


def create_experiment_manifest(
    project_dir: str | Path,
    repo_dir: str | Path,
    config: Dict[str, Any],
    config_filename: str = "run_config.json",
) -> Dict[str, Any]:
    project_dir = Path(project_dir)

    files = {
        "matrix": resolve_path(
            project_dir,
            config["matrix_file"],
        ),
        "taxonomy": resolve_path(
            project_dir,
            config["taxonomy_file"],
        ),
        "model_registry": resolve_path(
            project_dir,
            config["model_registry_file"],
        ),
        "generator_prompt": resolve_path(
            project_dir,
            config["generator_prompt_file"],
        ),
        "judge_prompt": resolve_path(
            project_dir,
            config["judge_prompt_file"],
        ),
        "config":
            project_dir / config_filename,
    }

    manifest = {
        "experiment_id":
            config["experiment_id"],
        "created_at_utc":
            utc_now(),
        "package_version":
            config["package_version"],
        "generator_prompt_version":
            config[
                "generator_prompt_version"
            ],
        "judge_prompt_version":
            config[
                "judge_prompt_version"
            ],
        "scenario_plan_version":
            config[
                "scenario_plan_version"
            ],
        "model_registry_version":
            config[
                "model_registry_version"
            ],
        "git_commit":
            git_commit_hash(repo_dir),
        "gpu":
            gpu_report(),
        "package_versions":
            package_versions(),
        "sha256": {
            name: sha256_file(path)
            for name, path in files.items()
            if path.exists()
        },
        "comparison_design": {
            "generator_families":
                config[
                    "generation_families"
                ],
            "judge_families":
                config[
                    "judge_families"
                ],
            "n_per_row":
                config["n_per_row"],
            "matrix_rows_requested":
                len(
                    config.get(
                        "matrix_ids"
                    ) or []
                ),
            "blind_judging": True,
            "python_decision_rule":
                config[
                    "decision_rule"
                ],
        },
    }

    out_path = resolve_path(
        project_dir,
        config[
            "experiment_manifest_file"
        ],
    )

    write_json(
        out_path,
        manifest,
    )

    return manifest
