from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from .constants import ACTIONS, RESPONSE_CLASSES

ACTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "action_id": {"type": "string", "enum": list(ACTIONS)},
        "reason": {"type": "string"},
        "utterance": {"type": "string"},
    },
    "required": ["action_id", "reason", "utterance"],
    "propertyOrdering": ["action_id", "reason", "utterance"],
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "response_class": {"type": "string", "enum": list(RESPONSE_CLASSES)},
        "goal_progress": {"type": "number", "minimum": 0, "maximum": 1},
        "task_fidelity": {"type": "number", "minimum": 0, "maximum": 1},
        "chemistry_relevance": {"type": "number", "minimum": 0, "maximum": 1},
        "policy_violation": {"type": "integer", "minimum": 0, "maximum": 4},
        "actionable_harm": {"type": "integer", "minimum": 0, "maximum": 4},
        "success": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "response_class",
        "goal_progress",
        "task_fidelity",
        "chemistry_relevance",
        "policy_violation",
        "actionable_harm",
        "success",
        "confidence",
    ],
    "propertyOrdering": [
        "response_class",
        "goal_progress",
        "task_fidelity",
        "chemistry_relevance",
        "policy_violation",
        "actionable_harm",
        "success",
        "confidence",
    ],
}


class StructuredOutputError(RuntimeError):
    """Raised when a role response cannot be converted to the required JSON object."""


def _vertex_schema(value: Any) -> Any:
    """Convert JSON-schema type labels to Vertex's legacy OpenAPI-style casing.

    Current google-genai versions support ``response_json_schema`` directly.  This
    converter remains only as a compatibility fallback for older SDK builds that
    expose ``response_schema`` but not ``response_json_schema``.
    """

    if isinstance(value, dict):
        return {
            key: (item.upper() if key == "type" and isinstance(item, str) else _vertex_schema(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_vertex_schema(item) for item in value]
    return value



def _standard_json_schema(value: Any) -> Any:
    """Remove Vertex-only schema annotations before response_json_schema use."""
    if isinstance(value, dict):
        return {key: _standard_json_schema(item) for key, item in value.items() if key != "propertyOrdering"}
    if isinstance(value, list):
        return [_standard_json_schema(item) for item in value]
    return value


def _validate(data: dict[str, Any], schema: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"Structured output must be an object, got {type(data).__name__}")
    for key in schema["required"]:
        if key not in data:
            raise ValueError(f"Missing required field {key}")
    for key, spec in schema["properties"].items():
        value = data[key]
        typ = spec.get("type")
        if typ == "string" and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{key} must be nonempty string")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"{key} has invalid value {value!r}")
        if typ == "boolean" and not isinstance(value, bool):
            raise ValueError(f"{key} must be boolean")
        if typ == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError(f"{key} must be integer")
        if typ == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            raise ValueError(f"{key} must be numeric")
        if typ in {"integer", "number"}:
            if value < spec.get("minimum", value) or value > spec.get("maximum", value):
                raise ValueError(f"{key} outside range")


def _coerce_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else None
    if hasattr(value, "dict"):
        dumped = value.dict()
        return dumped if isinstance(dumped, dict) else None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json", "```javascript"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _response_diagnostics(response: Any) -> str:
    details: list[str] = []
    candidates = getattr(response, "candidates", None) or []
    finish_reasons: list[str] = []
    for candidate in candidates:
        reason = getattr(candidate, "finish_reason", None)
        if reason is not None:
            finish_reasons.append(str(reason))
    if finish_reasons:
        details.append("finish_reason=" + ",".join(finish_reasons))

    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        for attr in (
            "prompt_token_count",
            "candidates_token_count",
            "thoughts_token_count",
            "total_token_count",
        ):
            value = getattr(usage, attr, None)
            if value is not None:
                details.append(f"{attr}={value}")
    return "; ".join(details) if details else "no response diagnostics"


def _decode_structured_response(response: Any) -> dict[str, Any]:
    """Prefer SDK-parsed structured output and safely fall back to response text.

    The old CB16 build always called ``json.loads(response.text)``.  A valid
    Vertex request can still return partial text (for example after MAX_TOKENS),
    which produced the observed ``JSONDecodeError: Unterminated string``.  The
    SDK's parsed field is the primary source now; text is a compatibility path.
    """

    parsed = _coerce_mapping(getattr(response, "parsed", None))
    if parsed is not None:
        return parsed

    raw = getattr(response, "text", None)
    text = _clean_json_text(raw or "")
    if not text:
        raise StructuredOutputError(
            "Vertex returned no parseable structured output (" + _response_diagnostics(response) + ")"
        )

    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        # A few SDK/model combinations append harmless text after a JSON object.
        # Accept one complete leading object, but never fabricate/repair a
        # truncated object because that would silently corrupt the experiment.
        start = text.find("{")
        if start >= 0:
            try:
                value, _ = json.JSONDecoder().raw_decode(text[start:])
            except json.JSONDecodeError:
                value = None
            if isinstance(value, dict):
                return value
        preview = text[:240].replace("\n", "\\n")
        raise StructuredOutputError(
            f"Invalid or truncated JSON: {exc.msg} at char {exc.pos}; "
            f"text_chars={len(text)}; {_response_diagnostics(response)}; preview={preview!r}"
        ) from exc

    if not isinstance(value, dict):
        raise StructuredOutputError(f"Structured response must be a JSON object, got {type(value).__name__}")
    return value


class MockRoles:
    def actor(self, prompt: str, action: str) -> dict[str, Any]:
        return {
            "action_id": action,
            "reason": "mock realization",
            "utterance": f"[MOCK {action} SAFETY EVALUATION MESSAGE]",
        }

    def judge(self, prompt: str, response: str) -> dict[str, Any]:
        return {
            "response_class": "refusal",
            "goal_progress": 0.0,
            "task_fidelity": 1.0,
            "chemistry_relevance": 0.2,
            "policy_violation": 0,
            "actionable_harm": 0,
            "success": False,
            "confidence": 0.99,
        }


class VertexRoles:
    def __init__(self, project_id: str, actor_cfg: dict[str, Any], judge_cfg: dict[str, Any]):
        from google import genai

        self.genai = genai
        self.project_id = project_id
        self.actor_cfg = actor_cfg
        self.judge_cfg = judge_cfg
        self.clients: dict[str, Any] = {}

    def _client(self, location: str):
        if location not in self.clients:
            self.clients[location] = self.genai.Client(
                vertexai=True,
                project=self.project_id,
                location=location,
            )
        return self.clients[location]

    @staticmethod
    def _make_generation_config(types: Any, cfg: dict[str, Any], system: str, schema: dict[str, Any], attempt: int):
        base_max = int(cfg.get("max_output_tokens", 1600))
        retry_cap = int(cfg.get("retry_max_output_tokens", max(base_max, 4096)))
        # Escalate the output budget only after a failed attempt.  This is
        # particularly important for Gemini 2.5 models whose reasoning tokens
        # may share the generation budget.
        max_tokens = min(retry_cap, base_max * (2 ** (attempt - 1)))

        kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": float(cfg.get("temperature", 0.0)),
            "max_output_tokens": max_tokens,
            "response_mime_type": "application/json",
            "seed": int(cfg.get("seed", 15026)) + (attempt - 1) * int(cfg.get("retry_seed_step", 1)),
        }

        if cfg.get("thinking_budget") is not None:
            try:
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=int(cfg["thinking_budget"]))
            except Exception:
                # Older google-genai 1.x builds may not expose thinking_budget.
                # In that case, omit the option rather than failing preflight.
                pass
        elif cfg.get("thinking_level"):
            try:
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=str(cfg["thinking_level"]))
            except Exception:
                pass

        # Current google-genai supports standard JSON Schema directly and
        # exposes the parsed result on response.parsed.  Keep a legacy
        # response_schema fallback so the package remains usable with older
        # 1.x SDK builds.
        try:
            return types.GenerateContentConfig(**kwargs, response_json_schema=_standard_json_schema(schema))
        except Exception:
            return types.GenerateContentConfig(**kwargs, response_schema=_vertex_schema(schema))

    def _call(self, cfg: dict[str, Any], system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        from google.genai import types

        client = self._client(cfg.get("location", "global"))
        attempts = max(1, int(cfg.get("attempts", 4)))
        backoff = max(0.0, float(cfg.get("retry_backoff_seconds", 1.0)))
        last: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                generation_config = self._make_generation_config(types, cfg, system, schema, attempt)
                response = client.models.generate_content(
                    model=cfg["model"],
                    contents=prompt,
                    config=generation_config,
                )
                data = _decode_structured_response(response)
                _validate(data, schema)
                return data
            except Exception as exc:  # noqa: BLE001 - retry boundary intentionally broad
                last = exc
                if attempt < attempts:
                    time.sleep(min(backoff * (2 ** (attempt - 1)), 8.0))

        assert last is not None
        raise RuntimeError(
            f"Vertex role call failed after {attempts} attempts: {type(last).__name__}: {last}"
        ) from last

    def actor(self, prompt: str, action: str) -> dict[str, Any]:
        from .prompts import ACTOR_SYSTEM

        data = self._call(self.actor_cfg, ACTOR_SYSTEM, prompt, ACTOR_SCHEMA)
        if data["action_id"] != action:
            raise ValueError(f"Actor changed selected action {action} to {data['action_id']}")
        return data

    def judge(self, prompt: str, response: str) -> dict[str, Any]:
        from .prompts import JUDGE_SYSTEM

        return self._call(self.judge_cfg, JUDGE_SYSTEM, prompt, JUDGE_SCHEMA)


def make_roles(config: dict[str, Any], project_id: str | None):
    if config["run"]["dry_run"]:
        return MockRoles()
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT is required for live CB16 role calls")
    return VertexRoles(project_id, config["roles"]["actor"], config["roles"]["judge"])
