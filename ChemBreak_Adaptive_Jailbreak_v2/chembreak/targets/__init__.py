from .chemdfm import ChemDFMSession
from .chemllm import ChemLLMSession
from .llasmol import LlaSMolSession


def build_target(name: str, spec: dict, generation: dict, runtime: dict | None = None):
    loader = spec.get("loader", name)
    runtime = runtime or {}
    if loader == "chemdfm":
        return ChemDFMSession(spec, generation, runtime)
    if loader == "chemllm":
        return ChemLLMSession(spec, generation, runtime)
    if loader == "llasmol":
        return LlaSMolSession(spec, generation, runtime)
    raise ValueError(f"Unknown target loader: {loader}")
