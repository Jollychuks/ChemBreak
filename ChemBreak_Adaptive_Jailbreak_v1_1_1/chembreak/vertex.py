from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import json
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


@dataclass
class VertexGeminiClient:
    model: str
    project: str
    location: str = "global"
    temperature: float = 0.0
    max_output_tokens: int = 1200

    def __post_init__(self) -> None:
        self.project = _detect_project(self.project)
        self.client = genai.Client(vertexai=True, project=self.project, location=self.location)

    def text(self, prompt: str, *, json_mode: bool = False, temperature: float | None = None) -> str:
        cfg = types.GenerateContentConfig(
            temperature=self.temperature if temperature is None else temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        resp = self.client.models.generate_content(model=self.model, contents=prompt, config=cfg)
        return resp.text or ""

    def json(self, prompt: str, retries: int = 2) -> Any:
        last = None
        for attempt in range(retries + 1):
            raw = self.text(prompt, json_mode=True, temperature=0.0 if attempt else None)
            try:
                return extract_json(raw)
            except Exception as e:
                last = e
                prompt = prompt + "\nReturn only valid JSON matching the requested schema."
        raise RuntimeError(f"JSON generation failed for {self.model}: {last}")


@dataclass
class VertexOpenAICompatClient:
    model: str
    project: str
    location: str
    temperature: float = 0.0
    max_output_tokens: int = 1200
    reasoning_effort: str | None = None
    timeout_seconds: int = 240

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
        headers = {
            "Authorization": f"Bearer {self._token()}",
            "Content-Type": "application/json; charset=utf-8",
        }
        resp = requests.post(self.endpoint, headers=headers, json=body, timeout=self.timeout_seconds)
        if not resp.ok:
            text = resp.text[:2000]
            raise RuntimeError(f"{self.model} HTTP {resp.status_code}: {text}")
        obj = resp.json()
        try:
            return obj["choices"][0]["message"].get("content", "") or ""
        except Exception as e:
            raise RuntimeError(f"Unexpected response shape from {self.model}: {json.dumps(obj)[:2000]}") from e

    def json(self, prompt: str, retries: int = 2) -> Any:
        last = None
        for attempt in range(retries + 1):
            raw = self.text(prompt, json_mode=True, temperature=0.0)
            try:
                return extract_json(raw)
            except Exception as e:
                last = e
                prompt = prompt + "\nReturn only a syntactically valid JSON object matching the requested schema."
        raise RuntimeError(f"JSON generation failed for {self.model}: {last}")


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
