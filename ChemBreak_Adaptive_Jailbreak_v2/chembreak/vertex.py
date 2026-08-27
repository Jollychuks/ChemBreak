from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import json
import random
import time
import requests
import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest
from google import genai
from google.genai import types
from .jsonutil import extract_json


def _detect_project(project: str) -> str:
    if project:
        return project
    _, detected = google.auth.default()
    if not detected:
        raise RuntimeError("No Google Cloud project found. Set gcp.project or GOOGLE_CLOUD_PROJECT.")
    return detected


def _sleep_backoff(attempt: int, base: float, maximum: float, *, retry_after: str | None = None) -> None:
    if retry_after:
        try:
            delay = float(retry_after)
        except Exception:
            delay = min(maximum, base * (2 ** max(attempt - 1, 0)))
    else:
        delay = min(maximum, base * (2 ** max(attempt - 1, 0)))
    delay += random.uniform(0.0, min(1.0, delay * 0.15))
    time.sleep(max(0.0, delay))


def _short(text: str, limit: int = 500) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "..."


@dataclass
class VertexGeminiClient:
    model: str
    project: str
    location: str = "global"
    temperature: float = 0.0
    max_output_tokens: int = 1200
    network_max_attempts: int = 5
    initial_backoff_seconds: float = 4.0
    max_backoff_seconds: float = 45.0
    json_max_attempts: int = 4
    min_interval_seconds: float = 0.0
    _last_request_started: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.project = _detect_project(self.project)
        self.client = genai.Client(vertexai=True, project=self.project, location=self.location)

    def _pace(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        wait = self.min_interval_seconds - (time.monotonic() - self._last_request_started)
        if wait > 0:
            time.sleep(wait)

    def text(self, prompt: str, *, json_mode: bool = False, temperature: float | None = None) -> str:
        cfg = types.GenerateContentConfig(
            temperature=self.temperature if temperature is None else temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        last: Exception | None = None
        for attempt in range(1, self.network_max_attempts + 1):
            try:
                self._pace()
                self._last_request_started = time.monotonic()
                resp = self.client.models.generate_content(model=self.model, contents=prompt, config=cfg)
                return resp.text or ""
            except Exception as e:
                last = e
                if attempt >= self.network_max_attempts:
                    break
                print(f"[MODEL RETRY] {self.model} request attempt {attempt}/{self.network_max_attempts} failed: {_short(str(e))}", flush=True)
                _sleep_backoff(attempt, self.initial_backoff_seconds, self.max_backoff_seconds)
        raise RuntimeError(f"{self.model} request failed after {self.network_max_attempts} attempts: {last}") from last

    def json(self, prompt: str, retries: int | None = None) -> Any:
        total = self.json_max_attempts if retries is None else max(1, int(retries) + 1)
        last: Exception | None = None
        last_raw = ""
        base_prompt = prompt
        for attempt in range(1, total + 1):
            request_prompt = base_prompt
            if attempt > 1:
                request_prompt += (
                    "\n\nYour previous answer could not be parsed as the required JSON. "
                    "Re-evaluate the supplied content without reproducing hazardous details. "
                    "Return only one syntactically valid JSON object matching the requested keys. "
                    "No markdown, preamble, or commentary outside the JSON object."
                )
            last_raw = self.text(request_prompt, json_mode=True, temperature=0.0 if attempt > 1 else None)
            try:
                obj = extract_json(last_raw)
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected a JSON object, got {type(obj).__name__}")
                return obj
            except Exception as e:
                last = e
                if attempt < total:
                    print(f"[JSON RETRY] {self.model} parse attempt {attempt}/{total} failed: {e}", flush=True)
        raise RuntimeError(
            f"JSON generation failed for {self.model} after {total} attempts: {last}. "
            f"Last output preview: {_short(last_raw)}"
        ) from last


@dataclass
class VertexOpenAICompatClient:
    model: str
    project: str
    location: str
    temperature: float = 0.0
    max_output_tokens: int = 1200
    reasoning_effort: str | None = None
    timeout_seconds: int = 240
    network_max_attempts: int = 6
    initial_backoff_seconds: float = 5.0
    max_backoff_seconds: float = 60.0
    json_max_attempts: int = 4
    min_interval_seconds: float = 2.0
    _last_request_started: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.project = _detect_project(self.project)
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self.credentials = creds

    @property
    def endpoint(self) -> str:
        if self.location == "global":
            host = "https://aiplatform.googleapis.com"
        else:
            host = f"https://{self.location}-aiplatform.googleapis.com"
        return f"{host}/v1/projects/{self.project}/locations/{self.location}/endpoints/openapi/chat/completions"

    def _token(self) -> str:
        if not self.credentials.valid or self.credentials.expired or not self.credentials.token:
            self.credentials.refresh(GoogleAuthRequest())
        return str(self.credentials.token)

    def _pace(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        wait = self.min_interval_seconds - (time.monotonic() - self._last_request_started)
        if wait > 0:
            time.sleep(wait)

    def text(self, prompt: str, *, json_mode: bool = False, temperature: float | None = None) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_output_tokens,
            "stream": False,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort

        retryable = {408, 409, 429, 500, 502, 503, 504}
        last_error: Exception | None = None
        last_text = ""
        for attempt in range(1, self.network_max_attempts + 1):
            try:
                self._pace()
                self._last_request_started = time.monotonic()
                headers = {
                    "Authorization": f"Bearer {self._token()}",
                    "Content-Type": "application/json; charset=utf-8",
                }
                resp = requests.post(self.endpoint, headers=headers, json=body, timeout=self.timeout_seconds)
                last_text = resp.text[:2000]
                if resp.ok:
                    obj = resp.json()
                    try:
                        return obj["choices"][0]["message"].get("content", "") or ""
                    except Exception as e:
                        raise RuntimeError(f"Unexpected response shape from {self.model}: {json.dumps(obj)[:2000]}") from e
                if resp.status_code not in retryable:
                    raise RuntimeError(f"{self.model} HTTP {resp.status_code}: {last_text}")
                if attempt >= self.network_max_attempts:
                    raise RuntimeError(f"{self.model} HTTP {resp.status_code} after {attempt} attempts: {last_text}")
                print(
                    f"[MODEL RETRY] {self.model} HTTP {resp.status_code} attempt {attempt}/{self.network_max_attempts}; backing off",
                    flush=True,
                )
                _sleep_backoff(
                    attempt,
                    self.initial_backoff_seconds,
                    self.max_backoff_seconds,
                    retry_after=resp.headers.get("Retry-After"),
                )
            except requests.RequestException as e:
                last_error = e
                if attempt >= self.network_max_attempts:
                    break
                print(f"[MODEL RETRY] {self.model} network attempt {attempt}/{self.network_max_attempts} failed: {_short(str(e))}", flush=True)
                _sleep_backoff(attempt, self.initial_backoff_seconds, self.max_backoff_seconds)
        if last_error:
            raise RuntimeError(f"{self.model} request failed after {self.network_max_attempts} attempts: {last_error}") from last_error
        raise RuntimeError(f"{self.model} request failed after {self.network_max_attempts} attempts: {last_text}")

    def json(self, prompt: str, retries: int | None = None) -> Any:
        total = self.json_max_attempts if retries is None else max(1, int(retries) + 1)
        last: Exception | None = None
        last_raw = ""
        base_prompt = prompt
        for attempt in range(1, total + 1):
            request_prompt = base_prompt
            if attempt > 1:
                request_prompt += (
                    "\n\nYour previous answer could not be parsed as the required JSON. "
                    "Return only one syntactically valid JSON object matching the requested keys. "
                    "Do not reproduce or expand any hazardous content. No markdown or commentary outside JSON."
                )
            last_raw = self.text(request_prompt, json_mode=True, temperature=0.0)
            try:
                obj = extract_json(last_raw)
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected a JSON object, got {type(obj).__name__}")
                return obj
            except Exception as e:
                last = e
                if attempt < total:
                    print(f"[JSON RETRY] {self.model} parse attempt {attempt}/{total} failed: {e}", flush=True)
        raise RuntimeError(
            f"JSON generation failed for {self.model} after {total} attempts: {last}. "
            f"Last output preview: {_short(last_raw)}"
        ) from last


def build_client(spec: dict[str, Any], gcp_cfg: dict[str, Any]):
    provider = spec.get("provider")
    project = gcp_cfg.get("project", "")
    location = spec.get("location") or gcp_cfg.get("location", "global")
    common = dict(
        model=spec["model"],
        project=project,
        location=location,
        temperature=float(spec.get("temperature", 0.0)),
        max_output_tokens=int(spec.get("max_output_tokens", 1200)),
        network_max_attempts=int(spec.get("network_max_attempts", 5)),
        initial_backoff_seconds=float(spec.get("initial_backoff_seconds", 4.0)),
        max_backoff_seconds=float(spec.get("max_backoff_seconds", 45.0)),
        json_max_attempts=int(spec.get("json_max_attempts", 4)),
        min_interval_seconds=float(spec.get("min_interval_seconds", 0.0)),
    )
    if provider == "vertex_gemini":
        return VertexGeminiClient(**common)
    if provider == "vertex_openai_compat":
        return VertexOpenAICompatClient(
            **common,
            reasoning_effort=spec.get("reasoning_effort"),
            timeout_seconds=int(spec.get("timeout_seconds", 240)),
        )
    raise ValueError(f"Unsupported model provider: {provider}")
