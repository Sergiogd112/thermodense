"""Structural tests for the standalone HASDM preparation CLI path."""

import ast
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts/causal_hasdm_saber_maunaloa.py"
)


def load_prepare_predicate():
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "prepare_hasdm_only_requested"
    )
    namespace: dict[str, object] = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(SCRIPT_PATH), "exec"),
        namespace,
    )
    return namespace["prepare_hasdm_only_requested"]


def test_prepare_hasdm_only_predicate_accepts_only_the_standalone_flag():
    predicate = load_prepare_predicate()

    assert predicate([]) is False
    assert predicate(["--prepare-hasdm-only"]) is True
    with pytest.raises(ValueError, match="Unsupported arguments"):
        predicate(["--prepare-hasdm-only", "--other"])


def test_prepare_hasdm_only_exits_after_hasdm_definitions_before_saber_loading():
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    body = tree.body
    hasdm_index = next(
        index
        for index, node in enumerate(body)
        if isinstance(node, ast.FunctionDef)
        and node.name == "load_hasdm_maunaloa_daily"
    )
    saber_index = next(
        index
        for index, node in enumerate(body)
        if isinstance(node, ast.FunctionDef)
        and node.name == "load_saber_maunaloa_daily"
    )
    prepare_exit = body[hasdm_index + 1]

    assert isinstance(prepare_exit, ast.If)
    assert hasdm_index < body.index(prepare_exit) < saber_index
    assert any(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "__name__"
        for node in ast.walk(prepare_exit.test)
    )
    calls = [node.func for node in ast.walk(prepare_exit) if isinstance(node, ast.Call)]
    assert any(
        isinstance(func, ast.Name) and func.id == "load_hasdm_maunaloa_daily"
        for func in calls
    )
    assert any(
        isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "SystemExit"
        for node in ast.walk(prepare_exit)
    )
    source = ast.get_source_segment(
        SCRIPT_PATH.read_text(encoding="utf-8"), prepare_exit
    )
    assert "hasdm_maunaloa_daily_long.parquet" in source
    assert "hasdm_maunaloa_daily_wide.parquet" in source
    assert "HASDM altitude summary" in source
