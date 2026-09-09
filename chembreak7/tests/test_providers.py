from chembreak7.providers import ROLE_SCHEMAS, _openai_schema


def test_open_model_schema_is_strict_and_removes_vertex_only_ordering():
    schema = _openai_schema(ROLE_SCHEMAS["safety_verifier"])
    assert schema["additionalProperties"] is False
    assert "propertyOrdering" not in schema
    assert set(schema["required"]) == set(schema["properties"])
