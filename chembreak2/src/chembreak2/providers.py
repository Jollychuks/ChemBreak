from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from .utils import retry_call


@dataclass(slots=True)
class GenerationResponse:
    text: str
    model: str
    provider: str
    usage: dict[str, Any]


class MockProvider:
    def generate(self, *, system: str, prompt: str, role: str) -> GenerationResponse:
        import json

        if role == "planner":
            data = {"action_id": "REPHRASE", "reason": "Deterministic mock policy choice."}
        elif role == "realizer":
            data = {"utterance": "[MOCK RED-TEAM UTTERANCE WITHHELD IN DRY RUN]"}
        elif role == "asset_single":
            data = {
                "attempts": [
                    {"attempt": i, "strategy": "mock", "utterance": f"[MOCK SINGLE ATTEMPT {i}]"}
                    for i in range(1, 6)
                ]
            }
        elif role == "asset_fixed":
            data = {
                "turns": [
                    {"turn": i, "action_id": "REPHRASE", "utterance": f"[MOCK FIXED TURN {i}]"}
                    for i in range(1, 6)
                ]
            }
        elif role == "safety_judge":
            data = {
                "response_class": "refusal",
                "success": False,
                "goal_progress": 0.0,
                "task_fidelity": 1.0,
                "policy_violation": 0,
                "actionable_harm": 0,
                "confidence": 0.95,
                "rationale": "Mock target refused the request.",
            }
        elif role == "chemistry_judge":
            data = {
                "chemical_relevance": 0.0,
                "scientific_correctness": 0.0,
                "operational_completeness": 0.0,
                "entity_match": False,
                "confidence": 0.95,
                "rationale": "No chemistry assistance was supplied.",
            }
        elif role == "adjudicator":
            data = {
                "success": False,
                "response_class": "refusal",
                "goal_progress": 0.0,
                "task_fidelity": 1.0,
                "chemical_relevance": 0.0,
                "scientific_correctness": 0.0,
                "confidence": 0.95,
                "rationale": "Mock adjudication confirms refusal.",
            }
        else:
            data = {"text": "mock"}
        return GenerationResponse(json.dumps(data), "mock", "mock", {})


class VertexGenAIProvider:
    def __init__(self, project_id: str, settings: dict[str, Any]):
        from google import genai

        self.settings = settings
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=settings["location"],
        )

    def generate(self, *, system: str, prompt: str, role: str) -> GenerationResponse:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=float(self.settings.get("temperature", 0.0)),
            max_output_tokens=int(self.settings.get("max_output_tokens", 1200)),
            response_mime_type="application/json",
            seed=int(self.settings.get("seed", 9032026)),
        )
        response = self.client.models.generate_content(
            model=self.settings["model"], contents=prompt, config=config
        )
        usage = {}
        if getattr(response, "usage_metadata", None):
            usage = response.usage_metadata.model_dump(exclude_none=True)
        return GenerationResponse(response.text or "", self.settings["model"], "vertex_genai", usage)


class VertexOpenAIProvider:
    def __init__(self, project_id: str, settings: dict[str, Any]):
        import google.auth
        import google.auth.transport.requests
        import openai

        self.settings = settings
        self.project_id = project_id
        self.credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self.auth_request = google.auth.transport.requests.Request()
        self.credentials.refresh(self.auth_request)
        location = settings["location"]
        base_url = (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/"
            f"locations/{location}/endpoints/openapi"
        )
        self.openai_module = openai
        self.client = openai.OpenAI(base_url=base_url, api_key=self.credentials.token)

    def _refresh_if_needed(self) -> None:
        if not self.credentials.valid or self.credentials.expired:
            self.credentials.refresh(self.auth_request)
            self.client.api_key = self.credentials.token

    def generate(self, *, system: str, prompt: str, role: str) -> GenerationResponse:
        self._refresh_if_needed()
        response = self.client.chat.completions.create(
            model=self.settings["model"],
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=float(self.settings.get("temperature", 0.0)),
            max_tokens=int(self.settings.get("max_output_tokens", 1200)),
            response_format={"type": "json_object"},
            seed=int(self.settings.get("seed", 9032026)),
        )
        usage = response.usage.model_dump(exclude_none=True) if response.usage else {}
        return GenerationResponse(
            response.choices[0].message.content or "",
            self.settings["model"],
            "vertex_openai",
            usage,
        )


class RoleClients:
    def __init__(self, config: dict[str, Any], project_id: str | None):
        self.config = config
        self.project_id = project_id
        self.clients: dict[str, Any] = {}
        self.call_history: list[dict[str, Any]] = []
        for role, settings in config["roles"].items():
            settings = {**settings, "seed": int(config["run"]["seed"])}
            if config["run"]["dry_run"]:
                self.clients[role] = MockProvider()
            elif settings["provider"] == "vertex_genai":
                self.clients[role] = VertexGenAIProvider(str(project_id), settings)
            elif settings["provider"] == "vertex_openai":
                self.clients[role] = VertexOpenAIProvider(str(project_id), settings)
            else:
                raise ValueError(f"Unsupported provider for {role}: {settings['provider']}")

    def call(self, role: str, prompt: str, system: str, call_role: str | None = None) -> GenerationResponse:
        retry = self.config["retries"]
        started = time.perf_counter()
        response = retry_call(
            lambda: self.clients[role].generate(system=system, prompt=prompt, role=call_role or role),
            attempts=int(retry["planner_or_judge_attempts"]),
            initial_backoff=float(retry["initial_backoff_seconds"]),
            maximum_backoff=float(retry["maximum_backoff_seconds"]),
        )
        self.call_history.append(
            {
                "role": role,
                "call_role": call_role or role,
                "provider": response.provider,
                "model": response.model,
                "latency_seconds": time.perf_counter() - started,
                "usage": response.usage,
            }
        )
        return response

    def drain_call_history(self) -> list[dict[str, Any]]:
        history = self.call_history
        self.call_history = []
        return history
