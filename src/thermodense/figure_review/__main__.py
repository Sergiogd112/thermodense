"""Serve the maintained figure-review workbench on localhost.

One command to run (from the repo root):

    uv run python -m thermodense.figure_review

Opens http://127.0.0.1:8124/ in the default browser.
Regenerate the sample figures first if figures/ is empty:

    uv run python -m thermodense.figure_review.make_figures
"""
from __future__ import annotations

import functools
import http.server
import json
import os
from pathlib import Path
import threading
from urllib.parse import urlsplit, urlunsplit
import webbrowser

HERE = Path(__file__).resolve().parent
PORT = int(os.environ.get("THERMODENSE_FIGURE_REVIEW_PORT", "8124"))
HOST = "127.0.0.1"
SCOPED_PATH = "/figure-review"
INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Thermodense figure review</title>
  <script>
    if (!location.pathname.endsWith("/")) {
      const canonical = new URL(location.href);
      canonical.pathname += "/";
      location.replace(canonical);
    }
  </script>
  <link rel="stylesheet" href="review.css">
</head>
<body>
  <div id="app" aria-live="polite"><p class="loading">Loading figure set…</p></div>
  <script type="module" src="review.js"></script>
</body>
</html>
"""


class WorkbenchHandler(http.server.SimpleHTTPRequestHandler):
    """Serve route-safe workbench assets without stale development caching."""

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _strip_scoped_path(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path == SCOPED_PATH:
            path = "/"
        elif parsed.path.startswith(f"{SCOPED_PATH}/"):
            path = parsed.path[len(SCOPED_PATH) :]
        else:
            return
        self.path = urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))

    def _serves_index(self) -> bool:
        return urlsplit(self.path).path == "/"

    def _send_index(self, *, include_body: bool) -> None:
        encoded = INDEX_HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        if include_body:
            self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        self._strip_scoped_path()
        if self._serves_index():
            self._send_index(include_body=True)
        else:
            super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler API
        self._strip_scoped_path()
        if self._serves_index():
            self._send_index(include_body=False)
        else:
            super().do_HEAD()


def ensure_figure_assets() -> None:
    """Regenerate ignored browser/publication artifacts when absent."""
    data_path = HERE / "data.json"
    try:
        figure_set = json.loads(data_path.read_text())
        paths = [
            HERE / relative
            for figure in figure_set["figures"]
            for relative in (figure["src"], figure["publicationSrc"])
        ]
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
        paths = []
    if paths and all(path.exists() for path in paths):
        return
    from thermodense.figure_review import make_figures

    make_figures.main()


def main() -> None:
    ensure_figure_assets()
    handler = functools.partial(WorkbenchHandler, directory=str(HERE))
    httpd = http.server.ThreadingHTTPServer((HOST, PORT), handler)
    url = f"http://{HOST}:{PORT}/"
    if os.environ.get("THERMODENSE_FIGURE_REVIEW_NO_OPEN") != "1":
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    print(f"Figure-review workbench -> {url}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
