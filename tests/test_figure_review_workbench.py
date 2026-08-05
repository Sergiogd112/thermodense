from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

from thermodense.figure_review.__main__ import HOST, INDEX_HTML, WorkbenchHandler


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_figure_review_core_javascript_contract() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    result = subprocess.run(
        [node, "--test", "tests/js/figure_review_core.test.mjs"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_workbench_assets_use_relative_routes() -> None:
    package = REPO_ROOT / "src/thermodense/figure_review"
    assert 'href="review.css"' in INDEX_HTML
    assert 'src="review.js"' in INDEX_HTML
    assert (package / "data.json").exists()


def test_server_is_localhost_only_and_strips_scoped_route() -> None:
    assert HOST == "127.0.0.1"
    handler = object.__new__(WorkbenchHandler)
    handler.path = "/figure-review/figures/figure.png?download=1"
    handler._strip_scoped_path()
    assert handler.path == "/figures/figure.png?download=1"
