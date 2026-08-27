from .chemdfm import ChemDFMSession
from .chemllm import ChemLLMSession
from .llasmol import LlaSMolSession


def build_target(name: str, spec: dict, generation: dict):
    loader = spec.get("loader", name)
    if loader == "chemdfm":
        return ChemDFMSession(spec, generation)
    if loader == "chemllm":
        return ChemLLMSession(spec, generation)
    if loader == "llasmol":
        return LlaSMolSession(spec, generation)
    raise ValueError(f"Unknown target loader: {loader}")
