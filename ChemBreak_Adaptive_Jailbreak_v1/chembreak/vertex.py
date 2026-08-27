from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import google.auth
from google import genai
from google.genai import types
from .jsonutil import extract_json


@dataclass
class VertexGeminiClient:
    model: str
    project: str
    location: str
    temperature: float = 0.0
    max_output_tokens: int = 1200

    def __post_init__(self) -> None:
        project = self.project
        if not project:
            _, detected = google.auth.default()
            project = detected or ""
        if not project:
            raise RuntimeError("No Google Cloud project found. Set gcp.project or GOOGLE_CLOUD_PROJECT.")
        self.project = project
        self.client = genai.Client(vertexai=True, project=project, location=self.location)

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
        raise RuntimeError(f"JSON generation failed: {last}")


def build_client(spec: dict[str, Any], gcp_cfg: dict[str, Any]) -> VertexGeminiClient:
    if spec.get("provider") != "vertex_gemini":
        raise ValueError(f"Unsupported provider in V1 package: {spec.get('provider')}")
    return VertexGeminiClient(
        model=spec["model"],
        project=gcp_cfg.get("project", ""),
        location=gcp_cfg.get("location", "global"),
        temperature=float(spec.get("temperature", 0.0)),
        max_output_tokens=int(spec.get("max_output_tokens", 1200)),
    )
