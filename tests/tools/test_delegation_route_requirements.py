"""Delegation providers receive capability requirements without raw image or schema data."""

from agent.delegation_provider import DelegationProvider
from tools.delegate_tool_routes import _route_one


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
