"""Delegation providers receive capability requirements without raw image or schema data."""

import json
from types import SimpleNamespace

from agent.delegation_provider import DelegationProvider
from tools.delegate_tool_routes import TaskRoute, _route_one


class RecordingProvider(DelegationProvider):
    name = "recording"

    def __init__(self):
        self.task = None

    def route_task(self, task, config):
        self.task = task
        return {"model": "gpt-6-luna"}


def test_route_receives_capabilities_without_image_or_schema_contents():
    provider = RecordingProvider()
    route = _route_one(provider, {}, 2, {
        "goal": "Describe screenshot", "context": "A test screen",
        "toolsets": ["terminal", "browser"],
        "images": [{"data": "private-image-bytes"}],
        "output_schema": {"secret": "not-for-router"},
        "role": "worker",
    }, {})

    assert route["model"] == "gpt-6-luna"
    assert provider.task == {
        "goal": "Describe screenshot", "context": "A test screen", "index": 2,
        "requirements": {
            "toolsets": ["terminal", "browser"],
            "has_images": True,
            "structured_output": True,
            "role": "worker",
        },
    }


def test_route_receives_empty_requirements_for_text_only_task():
    provider = RecordingProvider()
    _route_one(provider, {}, 0, {"goal": "Summarize"}, {})
    assert provider.task["requirements"] == {
        "toolsets": [], "has_images": False, "structured_output": False, "role": None,
    }


def test_route_uses_normalized_top_level_inputs_and_inherited_child_tools(monkeypatch):
    from tools import delegate_tool

    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    parent = SimpleNamespace(enabled_toolsets=["terminal", "browser"], disabled_toolsets=["browser"])
    provider = RecordingProvider()

    def capture_routes(tasks, config, credentials_cfg, agent):
        _route_one(provider, {}, 0, tasks[0], config)
        return [TaskRoute({}, {}, error="intentional test stop")], None

    monkeypatch.setattr(delegate_tool, "_routes_for", capture_routes)
    result = delegate_tool.delegate_task(
        goal="Describe screenshot", output_schema=schema,
        images=["data:image/png;base64,cHJpdmF0ZQ=="], parent_agent=parent,
    )

    assert "intentional test stop" in json.loads(result)["error"]
    assert provider.task["requirements"] == {
        "toolsets": ["terminal"], "has_images": True, "structured_output": True, "role": "leaf",
    }
    assert "images" not in provider.task and "output_schema" not in provider.task
