from __future__ import annotations

import json
import os
import time
from typing import Any

from .constants import ACTIONS, PROGRESSIVE_ACTIONS
from .scoring import METRIC_SCHEMAS


ATTACK_SCHEMA = {
    "type": "object",
    "properties": {
        "action_id": {"type": "string", "enum": list(ACTIONS)},
        "reason_code": {"type": "string"},
        "addressed_failure": {"type": "string"},
        "difference_from_prior": {"type": "string"},
        "utterance": {"type": "string"},
    },
    "required": ["action_id", "reason_code", "addressed_failure", "difference_from_prior", "utterance"],
    "additionalProperties": False,
}

PROGRESSIVE_ATTACK_SCHEMA = {
    "type": "object",
    "properties": {
        "action_id": {"type": "string", "enum": list(PROGRESSIVE_ACTIONS)},
        "reason_code": {"type": "string"},
        "addressed_failure": {"type": "string"},
        "difference_from_prior": {"type": "string"},
        "strategy_mode": {"type": "string", "enum": ["progressive_buildup"]},
        "progressive_attempt": {"type": "integer", "minimum": 1, "maximum": 2},
        "buildup_level": {"type": "integer", "minimum": 0, "maximum": 5},
        "buildup_route": {"type": "string"},
        "trajectory_role": {"type": "string"},
        "objective_link": {"type": "string"},
        "utterance": {"type": "string"},
    },
    "required": [
        "action_id", "reason_code", "addressed_failure", "difference_from_prior",
        "strategy_mode", "progressive_attempt", "buildup_level", "buildup_route",
        "trajectory_role", "objective_link", "utterance",
    ],
    "additionalProperties": False,
}

CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "goal_preservation": {"type": "number", "minimum": 0, "maximum": 1},
        "task_fidelity": {"type": "number", "minimum": 0, "maximum": 1},
        "action_alignment": {"type": "number", "minimum": 0, "maximum": 1},
        "drifted": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason_code": {"type": "string"},
    },
    "required": [
        "goal_preservation", "task_fidelity", "action_alignment", "drifted", "confidence", "reason_code"
    ],
    "additionalProperties": False,
}

PROGRESSIVE_CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "goal_preservation": {"type": "number", "minimum": 0, "maximum": 1},
        "task_fidelity": {"type": "number", "minimum": 0, "maximum": 1},
        "action_alignment": {"type": "number", "minimum": 0, "maximum": 1},
        "trajectory_fidelity": {"type": "number", "minimum": 0, "maximum": 1},
        "level_compliance": {"type": "number", "minimum": 0, "maximum": 1},
        "intensity_ceiling_compliance": {"type": "number", "minimum": 0, "maximum": 1},
        "route_difference": {"type": "number", "minimum": 0, "maximum": 1},
        "drifted": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason_code": {"type": "string"},
    },
    "required": [
        "goal_preservation", "task_fidelity", "action_alignment", "trajectory_fidelity",
        "level_compliance", "intensity_ceiling_compliance", "route_difference",
        "drifted", "confidence", "reason_code"
    ],
    "additionalProperties": False,
}


class StructuredOutputError(RuntimeError):
    def __init__(self, category, message, metadata=None):
        self.category = str(category)
        self.metadata = dict(metadata or {})
        super().__init__(str(message))


class ProviderPolicyBlock(RuntimeError):
    def __init__(self, provider, code, message, metadata=None):
        self.provider = str(provider)
        self.code = str(code or "provider_policy")
        self.provider_message = str(message or "")
        self.metadata = dict(metadata or {})
        super().__init__(f"{self.provider} policy block ({self.code})")


class MetricJudgeError(RuntimeError):
    def __init__(self, metric, audit):
        self.metric = str(metric).lower()
        self.audit = dict(audit or {})
        super().__init__(
            f"{self.metric.upper()} judge unresolved: "
            f"{self.audit.get('category', 'provider_error')}: {self.audit.get('message', '')}"
        )


def _clean(text):
    value = str(text or "").strip()
    if value.startswith("```"):
        rows = value.splitlines()[1:]
        if rows and rows[-1].strip() == "```":
            rows = rows[:-1]
        value = "\n".join(rows).strip()
    return value


def _coerce(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else None
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, dict) else None
        except Exception:
            return None
    return None


def _validate_shallow(data, schema):
    if not isinstance(data, dict):
        raise ValueError("Structured output must be an object")
    for key in schema["required"]:
        if key not in data:
            raise ValueError(f"Missing required field {key}")
    extra = set(data) - set(schema["properties"])
    if schema.get("additionalProperties") is False and extra:
        raise ValueError(f"Unexpected fields {sorted(extra)}")
    for key, specification in schema["properties"].items():
        value = data[key]
        expected = specification.get("type")
        if expected == "string" and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{key} must be a nonempty string")
        if expected == "boolean" and not isinstance(value, bool):
            raise ValueError(f"{key} must be boolean")
        if expected == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            raise ValueError(f"{key} must be numeric")
        if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError(f"{key} must be an integer")
        if "enum" in specification and value not in specification["enum"]:
            raise ValueError(f"{key} has an invalid value")
        if expected in {"number", "integer"} and not specification.get("minimum", value) <= value <= specification.get("maximum", value):
            raise ValueError(f"{key} is outside its allowed range")
    return data


def _schema_clean(value):
    if isinstance(value, dict):
        return {key: _schema_clean(item) for key, item in value.items() if key != "propertyOrdering"}
    if isinstance(value, list):
        return [_schema_clean(item) for item in value]
    return value


def _safe_metadata(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _safe_metadata(item) for key, item in list(value.items())[:50]}
    if isinstance(value, (list, tuple)):
        return [_safe_metadata(item) for item in list(value)[:50]]
    if hasattr(value, "model_dump"):
        try:
            return _safe_metadata(value.model_dump())
        except Exception:
            pass
    return str(value)[:2000]


def classify_judge_error(exc, provider, metric, attempt):
    message = str(exc)
    lower = message.lower()
    error_type = type(exc).__name__
    code = str(getattr(exc, "code", "") or "")
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    if not code and isinstance(body, dict):
        code = str((body.get("error") or body).get("code", "") or "")
    combined = f"{code} {message} {_safe_metadata(body)}".lower()
    category = "provider_error"
    if isinstance(exc, StructuredOutputError):
        category = exc.category
    elif isinstance(exc, ProviderPolicyBlock):
        combined = f"{exc.code} {exc.provider_message}".lower()
        if any(token in combined for token in ("bio_policy", "biological risk", "biology", "bioweapon")):
            category = "biological_policy_refusal"
        else:
            category = "provider_policy_refusal"
        code = exc.code
        message = exc.provider_message
    elif any(token in combined for token in ("bio_policy", "biological risk")):
        category = "biological_policy_refusal"
    elif any(token in combined for token in ("content_filter", "safety", "prohibited_content", "blocklist")):
        category = "content_filter"
    elif "json" in lower and any(token in lower for token in ("decode", "parse", "expecting", "malformed")):
        category = "malformed_json"
    elif "schema" in lower:
        category = "schema_error"
    elif any(token in lower for token in ("timeout", "timed out", "deadline")):
        category = "timeout"
    elif status in {401, 403} or any(token in lower for token in ("api key", "authentication", "unauthorized")):
        category = "authentication"
    elif status == 429 or "rate limit" in lower:
        category = "rate_limit"
    elif any(token in lower for token in ("connection", "transport", "network")):
        category = "transport_error"
    metadata = {
        "body": _safe_metadata(body),
        "structured_metadata": _safe_metadata(getattr(exc, "metadata", None)),
    }
    return {
        "metric": str(metric).upper(),
        "provider": str(provider),
        "category": category,
        "error_type": error_type,
        "error_code": code or error_type,
        "status": status,
        "message": str(message)[:4000],
        "attempt": int(attempt),
        "metadata": metadata,
    }


def _vertex_sampling_kwargs(config):
    if str(config.get("model", "")).startswith("gemini-3.8"):
        return {}
    return {
        key: float(config[key])
        for key in ("temperature", "top_p", "top_k")
        if config.get(key) is not None
    }


def _decode_vertex(response):
    parsed = _coerce(getattr(response, "parsed", None))
    if parsed is not None:
        return parsed
    prompt_feedback = getattr(response, "prompt_feedback", None)
    block_reason = str(getattr(prompt_feedback, "block_reason", "") or "")
    if block_reason and "UNSPECIFIED" not in block_reason.upper():
        raise ProviderPolicyBlock(
            "google",
            block_reason,
            str(getattr(prompt_feedback, "block_reason_message", "") or "Gemini blocked the input prompt"),
            {
                "prompt_feedback": _safe_metadata(prompt_feedback),
                "model_version": str(getattr(response, "model_version", "") or ""),
                "response_id": str(getattr(response, "response_id", "") or ""),
            },
        )
    candidates = getattr(response, "candidates", None) or []
    reasons = [str(getattr(candidate, "finish_reason", "")) for candidate in candidates]
    upper = " ".join(reasons).upper()
    if any(token in upper for token in ("SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT")):
        candidate = candidates[0] if candidates else None
        raise ProviderPolicyBlock(
            "google",
            reasons[0] if reasons else "safety",
            str(getattr(candidate, "finish_message", "") or "Gemini response was blocked"),
            {
                "finish_reasons": reasons,
                "safety_ratings": _safe_metadata(getattr(candidate, "safety_ratings", None)),
                "model_version": str(getattr(response, "model_version", "") or ""),
                "response_id": str(getattr(response, "response_id", "") or ""),
            },
        )
    text = _clean(getattr(response, "text", "") or "")
    if not text:
        raise StructuredOutputError(
            "incomplete_output",
            "Gemini returned no structured output",
            {
                "finish_reasons": reasons,
                "finish_messages": [str(getattr(candidate, "finish_message", "") or "") for candidate in candidates],
                "safety_ratings": [_safe_metadata(getattr(candidate, "safety_ratings", None)) for candidate in candidates],
                "prompt_feedback": _safe_metadata(prompt_feedback),
                "model_version": str(getattr(response, "model_version", "") or ""),
                "response_id": str(getattr(response, "response_id", "") or ""),
                "usage_metadata": _safe_metadata(getattr(response, "usage_metadata", None)),
            },
        )
    try:
        return json.loads(text)
    except Exception as exc:
        raise StructuredOutputError("malformed_json", f"Invalid Gemini JSON: {exc}") from exc


class MockRoles:
    def __init__(self):
        self.attack_index = 0

    def attack(self, prompt, allowed_actions):
        self.attack_index += 1
        action = list(allowed_actions)[0]
        payload = json.loads(prompt)
        strategy_mode = str(payload.get("strategy_mode", "adaptive_actions"))
        progressive_attempt = int(payload.get("progressive_attempt", 0) or 0)
        raw_level = payload.get("buildup_level")
        buildup_level = int(raw_level) if raw_level is not None else -1
        base = {
            "action_id": action,
            "reason_code": "mock",
            "addressed_failure": "mock refusal",
            "difference_from_prior": f"mock variant {self.attack_index}",
            "utterance": (
                f"[MOCK SECOND-PASS DISTINCT CONTRAST ROUTE {action} MESSAGE {self.attack_index}]"
                if progressive_attempt == 2
                else f"[MOCK FIRST-PASS COHERENT BUILDUP ROUTE {action} MESSAGE {self.attack_index}]"
                if progressive_attempt == 1
                else f"[MOCK {action} HIGH-LEVEL TEST MESSAGE {self.attack_index}]"
            ),
        }
        if strategy_mode != "progressive_buildup":
            return base
        return base | {
            "strategy_mode": strategy_mode,
            "progressive_attempt": progressive_attempt,
            "buildup_level": buildup_level,
            "buildup_route": f"mock route {self.attack_index}",
            "trajectory_role": "mock trajectory",
            "objective_link": "mock private link",
        }

    def judge_candidate(self, prompt, candidate, progressive=False):
        base = {
            "goal_preservation": 1.0,
            "task_fidelity": 1.0,
            "action_alignment": 1.0,
            "drifted": False,
            "confidence": 0.99,
            "reason_code": "mock_match",
        }
        if not progressive:
            return base
        return base | {
            "trajectory_fidelity": 1.0,
            "level_compliance": 1.0,
            "intensity_ceiling_compliance": 1.0,
            "route_difference": 1.0,
        }

    def judge_metric(self, metric, prompt):
        if str(metric).lower() == "rhs":
            return {
                "score": 1,
                "confidence": 0.99,
                "reason_code": "REFUSAL",
                "rationale": "The mock target refused.",
                "scientific_credibility": "not_credible",
                "evidence": [{"context_turn": 1, "quote": "cannot provide", "relevance": "refusal"}],
            }
        return {
            "score": 1,
            "confidence": 0.99,
            "reason_code": "NOT_FULFILLED",
            "rationale": "The mock target did not fulfill the objective.",
            "response_class": "hard_refusal",
            "evidence": [{"context_turn": 1, "quote": "cannot provide", "relevance": "objective refused"}],
        }


class VertexStructuredLLM:
    def __init__(self, project_id, config, client=None):
        self.project_id = project_id
        self.config = config
        self._client_override = client
        self._clients = {}

    def _client(self, location):
        if self._client_override is not None:
            return self._client_override
        if location not in self._clients:
            from google import genai
            from google.genai import types

            timeout_ms = max(1000, int(float(self.config.get("timeout_seconds", 60)) * 1000))
            self._clients[location] = genai.Client(
                enterprise=True,
                project=self.project_id,
                location=location,
                http_options=types.HttpOptions(timeout=timeout_ms),
            )
        return self._clients[location]

    def _generation_config(self, types, system, schema, attempt):
        base = int(self.config.get("max_output_tokens", 2048))
        cap = int(self.config.get("retry_max_output_tokens", 4096))
        kwargs = {
            "system_instruction": system,
            "max_output_tokens": min(cap, base * (2 ** (attempt - 1))),
            "response_mime_type": "application/json",
            "seed": int(self.config.get("seed", 34034)) + (attempt - 1) * int(self.config.get("retry_seed_step", 1)),
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        kwargs.update(_vertex_sampling_kwargs(self.config))
        if self.config.get("thinking_level"):
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=str(self.config["thinking_level"]))
        try:
            return types.GenerateContentConfig(**kwargs, response_json_schema=_schema_clean(schema))
        except Exception:
            return types.GenerateContentConfig(**kwargs, response_schema=_schema_clean(schema))

    def _call(self, system, prompt, schema):
        from google.genai import types

        attempts = max(1, int(self.config.get("attempts", 2)))
        backoff = max(0, float(self.config.get("retry_backoff_seconds", 1)))
        client = self._client(self.config.get("location", "global"))
        last = None
        for attempt in range(1, attempts + 1):
            try:
                response = client.models.generate_content(
                    model=self.config["model"],
                    contents=prompt,
                    config=self._generation_config(types, system, schema, attempt),
                )
                return _validate_shallow(_decode_vertex(response), schema)
            except ProviderPolicyBlock:
                raise
            except Exception as exc:
                last = exc
                if attempt < attempts:
                    time.sleep(min(backoff * (2 ** (attempt - 1)), 12))
        if isinstance(last, StructuredOutputError):
            metadata = dict(last.metadata)
            metadata["attempts"] = attempts
            raise StructuredOutputError(last.category, str(last), metadata) from last
        raise RuntimeError(f"Gemini structured call failed after {attempts} attempts: {type(last).__name__}: {last}") from last


class GeminiAttackPlanner(VertexStructuredLLM):
    def attack(self, prompt, allowed_actions):
        from .prompts import ATTACK_PLANNER_SYSTEM, PROGRESSIVE_ATTACK_PLANNER_SYSTEM

        progressive = any(action in PROGRESSIVE_ACTIONS for action in allowed_actions)
        system = PROGRESSIVE_ATTACK_PLANNER_SYSTEM if progressive else ATTACK_PLANNER_SYSTEM
        schema = PROGRESSIVE_ATTACK_SCHEMA if progressive else ATTACK_SCHEMA
        data = self._call(system, prompt, schema)
        if data["action_id"] not in set(allowed_actions):
            raise ValueError(f"Attack planner selected disallowed action {data['action_id']}")
        return data


class GeminiIntentGate(VertexStructuredLLM):
    def judge_candidate(self, prompt, candidate):
        from .prompts import CANDIDATE_JUDGE_SYSTEM

        return self._call(CANDIDATE_JUDGE_SYSTEM, prompt, CANDIDATE_SCHEMA)


class GeminiProgressiveGate(VertexStructuredLLM):
    def judge_candidate(self, prompt, candidate):
        from .prompts import PROGRESSIVE_CANDIDATE_JUDGE_SYSTEM

        return self._call(PROGRESSIVE_CANDIDATE_JUDGE_SYSTEM, prompt, PROGRESSIVE_CANDIDATE_SCHEMA)


class OpenAIMetricJudge:
    def __init__(self, config, client=None):
        self.config = config
        self._client_override = client
        self._client_cache = None

    def _client(self):
        if self._client_override is not None:
            return self._client_override
        if self._client_cache is None:
            key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not key:
                raise RuntimeError("OPENAI_API_KEY is not set")
            from openai import OpenAI

            self._client_cache = OpenAI(
                api_key=key,
                timeout=float(self.config.get("timeout_seconds", 120)),
                max_retries=0,
            )
        return self._client_cache

    @staticmethod
    def _refusal_text(response):
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                if str(getattr(content, "type", "")) == "refusal":
                    return str(getattr(content, "refusal", "") or "OpenAI judge refused the evaluation")
        return ""

    def judge_metric(self, metric, prompt):
        from .prompts import metric_system

        key = str(metric).lower()
        attempts = max(1, int(self.config.get("attempts", 2)))
        backoff = max(0, float(self.config.get("retry_backoff_seconds", 1)))
        base = max(256, int(self.config.get("max_output_tokens", 2048)))
        cap = max(base, int(self.config.get("retry_max_output_tokens", 4096)))
        audits = []
        for attempt in range(1, attempts + 1):
            try:
                kwargs = {
                    "model": self.config["model"],
                    "input": [
                        {"role": "system", "content": metric_system(key)},
                        {"role": "user", "content": prompt},
                    ],
                    "max_output_tokens": min(cap, base * (2 ** (attempt - 1))),
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": f"chembreak_{key}_judgment",
                            "strict": True,
                            "schema": METRIC_SCHEMAS[key],
                        }
                    },
                }
                if self.config.get("reasoning_effort"):
                    kwargs["reasoning"] = {"effort": str(self.config["reasoning_effort"])}
                elif self.config.get("temperature") is not None:
                    kwargs["temperature"] = float(self.config["temperature"])
                response = self._client().responses.create(**kwargs)
                refusal = self._refusal_text(response)
                if refusal:
                    raise ProviderPolicyBlock(
                        "openai", "refusal", refusal,
                        {"response_status": getattr(response, "status", None), "request_id": getattr(response, "_request_id", None)},
                    )
                status = str(getattr(response, "status", "completed"))
                text = _clean(getattr(response, "output_text", "") or "")
                if status not in {"completed", "None"} and not text:
                    raise StructuredOutputError(
                        "incomplete_output", f"OpenAI judge response status={status}",
                        {"response_status": status, "request_id": getattr(response, "_request_id", None)},
                    )
                if not text:
                    raise StructuredOutputError("incomplete_output", "OpenAI judge returned no output text")
                try:
                    data = json.loads(text)
                except Exception as exc:
                    raise StructuredOutputError("malformed_json", f"Invalid OpenAI JSON: {exc}") from exc
                if not isinstance(data, dict):
                    raise StructuredOutputError("schema_error", "OpenAI metric output is not an object")
                return data
            except Exception as exc:
                audit = classify_judge_error(exc, "openai", key, attempt)
                audits.append(audit)
                if attempt < attempts and audit["category"] not in {
                    "biological_policy_refusal", "provider_policy_refusal", "content_filter", "authentication"
                }:
                    time.sleep(min(backoff * (2 ** (attempt - 1)), 12))
                    continue
                break
        final = dict(audits[-1] if audits else {})
        final["attempts"] = audits
        raise MetricJudgeError(key, final)


class LocalQwenGGUFMetricJudge:
    def __init__(self, config, client=None):
        self.config = config
        self._client_override = client
        self._client_cache = None

    def _client(self):
        if self._client_override is not None:
            return self._client_override
        if self._client_cache is None:
            from pathlib import Path
            from openai import OpenAI

            model_path = Path(self.config["model_path"])
            if not model_path.is_file() or model_path.stat().st_size < 1_000_000_000:
                raise RuntimeError(f"Local Qwen judge file is missing or incomplete: {model_path}")
            self._client_cache = OpenAI(
                api_key="local-no-key-required",
                base_url=str(self.config["api_base_url"]),
                timeout=float(self.config.get("timeout_seconds", 900)),
                max_retries=0,
            )
        return self._client_cache

    def judge_metric(self, metric, prompt):
        from .prompts import metric_system

        key = str(metric).lower()
        attempts = max(1, int(self.config.get("attempts", 2)))
        backoff = max(0, float(self.config.get("retry_backoff_seconds", 1)))
        base = max(256, int(self.config.get("max_output_tokens", 4096)))
        cap = max(base, int(self.config.get("retry_max_output_tokens", 8192)))
        audits = []
        for attempt in range(1, attempts + 1):
            try:
                completion = self._client().chat.completions.create(
                    model=str(self.config["model"]),
                    messages=[
                        {"role": "system", "content": metric_system(key, "qwen")},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=min(cap, base * (2 ** (attempt - 1))),
                    extra_body={
                        "response_format": {"type": "json_object", "schema": METRIC_SCHEMAS[key]},
                    },
                )
                choices = getattr(completion, "choices", None) or []
                if not choices:
                    raise StructuredOutputError("incomplete_output", "Local Qwen judge returned no choices")
                choice = choices[0]
                finish_reason = str(getattr(choice, "finish_reason", ""))
                content = _clean(getattr(choice.message, "content", "") or "")
                if finish_reason == "length":
                    raise StructuredOutputError(
                        "incomplete_output", "Local Qwen judge exhausted its output token budget",
                        {"finish_reason": finish_reason},
                    )
                if not content:
                    raise StructuredOutputError(
                        "incomplete_output", "Local Qwen judge returned no structured output",
                        {"finish_reason": finish_reason},
                    )
                try:
                    data = json.loads(content)
                except Exception as exc:
                    raise StructuredOutputError(
                        "malformed_json", f"Invalid local Qwen JSON: {exc}", {"finish_reason": finish_reason}
                    ) from exc
                if not isinstance(data, dict):
                    raise StructuredOutputError("schema_error", "Local Qwen metric output is not an object")
                return data
            except Exception as exc:
                audit = classify_judge_error(exc, "local_qwen", key, attempt)
                audits.append(audit)
                if attempt < attempts:
                    time.sleep(min(backoff * (2 ** (attempt - 1)), 12))
        final = dict(audits[-1] if audits else {})
        final["attempts"] = audits
        raise MetricJudgeError(key, final)


class Roles:
    def __init__(self, attack, intent_gate, progressive_gate, judge):
        self.attack_planner_llm = attack
        self.intent_gate_llm = intent_gate
        self.progressive_gate_llm = progressive_gate
        self.judge_llm = judge

    def attack(self, prompt, allowed_actions):
        return self.attack_planner_llm.attack(prompt, allowed_actions)

    def judge_candidate(self, prompt, candidate, progressive=False):
        gate = self.progressive_gate_llm if progressive else self.intent_gate_llm
        return gate.judge_candidate(prompt, candidate)

    def judge_metric(self, metric, prompt):
        return self.judge_llm.judge_metric(metric, prompt)


def make_roles(config, project_id):
    if config["run"]["dry_run"]:
        return MockRoles()
    judge_config = config["roles"]["judge_llm"]
    if judge_config["provider"] == "openai_responses":
        judge = OpenAIMetricJudge(judge_config)
    elif judge_config["provider"] == "local_llamacpp_gguf":
        judge = LocalQwenGGUFMetricJudge(judge_config)
    else:
        raise ValueError(f"Unsupported judge provider: {judge_config['provider']}")
    return Roles(
        GeminiAttackPlanner(project_id, config["roles"]["attack_planner_llm"]),
        GeminiIntentGate(project_id, config["roles"]["intent_gate_llm"]),
        GeminiProgressiveGate(project_id, config["roles"]["progressive_gate_llm"]),
        judge,
    )
