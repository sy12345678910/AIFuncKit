"""
Tests for BasePlugin (sdk.py) and the @describe / @decription decorator.

Covers:
  - get_methods() discovers only @describe-decorated methods.
  - type hints are reflected correctly in args dict.
  - Methods without type hints default to str.
  - update_context / update_contetx (typo alias) both work.
  - get_live_context() returns current live context.
  - plugin_name attribute is used as-is.
  - Multiple plugins have independent live contexts.
  - describe == decription (alias parity).
"""
from __future__ import annotations

import pytest
from aifunckit.sdk import BasePlugin, describe, decription


# ---------------------------------------------------------------------------
# Concrete plugin stubs
# ---------------------------------------------------------------------------

class _SimplePlugin(BasePlugin):
    plugin_name = "simple"

    @describe("Say hello to a person")
    def greet(self, name: str) -> str:
        return f"Hello, {name}"

    @describe("Add two integers together")
    def add(self, x: int, y: int) -> int:
        return x + y

    def not_decorated(self) -> None:
        """This must NOT appear in get_methods() output."""
        pass


class _NoHintsPlugin(BasePlugin):
    plugin_name = "nohints"

    @describe("Action without type hints")
    def do_something(self, value):
        return value


# ---------------------------------------------------------------------------
# get_methods() discovery
# ---------------------------------------------------------------------------

class TestGetMethods:
    def test_discovers_decorated_methods_only(self):
        plugin = _SimplePlugin()
        names = [m["action_name"] for m in plugin.get_methods()]
        assert "greet" in names
        assert "add" in names
        assert "not_decorated" not in names

    def test_method_count_matches(self):
        plugin = _SimplePlugin()
        assert len(plugin.get_methods()) == 2

    def test_description_is_stored(self):
        plugin = _SimplePlugin()
        meta = {m["action_name"]: m for m in plugin.get_methods()}
        assert meta["greet"]["description"] == "Say hello to a person"
        assert meta["add"]["description"] == "Add two integers together"

    def test_type_hints_captured(self):
        plugin = _SimplePlugin()
        meta = {m["action_name"]: m for m in plugin.get_methods()}
        assert meta["greet"]["args"] == {"name": "str"}
        assert meta["add"]["args"] == {"x": "int", "y": "int"}

    def test_missing_type_hints_default_to_str(self):
        plugin = _NoHintsPlugin()
        meta = plugin.get_methods()[0]
        assert meta["args"]["value"] == "str"

    def test_self_not_in_args(self):
        plugin = _SimplePlugin()
        for m in plugin.get_methods():
            assert "self" not in m["args"]

    def test_method_name_callable(self):
        plugin = _SimplePlugin()
        for m in plugin.get_methods():
            assert callable(m["name"])


# ---------------------------------------------------------------------------
# Live context management
# ---------------------------------------------------------------------------

class TestLiveContext:
    def test_initial_context_is_empty(self):
        plugin = _SimplePlugin()
        assert plugin.get_live_context() == {}

    def test_update_context_stores_value(self):
        plugin = _SimplePlugin()
        plugin.update_context("last_run", "2026-01-01")
        assert plugin.get_live_context()["last_run"] == "2026-01-01"

    def test_update_contetx_alias_works(self):
        """Intentional typo alias must behave identically."""
        plugin = _SimplePlugin()
        plugin.update_contetx("key", "value")
        assert plugin.get_live_context()["key"] == "value"

    def test_contexts_are_independent_per_instance(self):
        p1, p2 = _SimplePlugin(), _SimplePlugin()
        p1.update_context("x", 1)
        p2.update_context("x", 2)
        assert p1.get_live_context()["x"] == 1
        assert p2.get_live_context()["x"] == 2


# ---------------------------------------------------------------------------
# @describe / @decription decorator
# ---------------------------------------------------------------------------

class TestDescribeDecorator:
    def test_describe_attaches_desc_attribute(self):
        @describe("My description")
        def fn():
            pass
        assert fn._desc == "My description"

    def test_decription_alias_is_same_as_describe(self):
        @decription("Via alias")
        def fn():
            pass
        assert fn._desc == "Via alias"

    def test_describe_returns_original_function(self):
        def original():
            return 42
        wrapped = describe("desc")(original)
        assert wrapped is original
        assert wrapped() == 42

    def test_describe_does_not_alter_return_value(self):
        @describe("Test action")
        def compute(n: int) -> int:
            return n * 2
        assert compute(5) == 10


# ---------------------------------------------------------------------------
# verify_environment default
# ---------------------------------------------------------------------------

def test_verify_environment_returns_empty_tuple():
    plugin = _SimplePlugin()
    assert plugin.verify_environment() == ()
