"""PROTOTYPE — serve the figure-review workbench on localhost.

One command to run (from the repo root):

    PYTHONPATH=src python -m thermodense.prototypes.figure_review

Opens http://127.0.0.1:8124/?variant=B in the default browser.
Regenerate the sample figures first if figures/ is empty:

    PYTHONPATH=src python -m thermodense.prototypes.figure_review.make_figures
"""
from __future__ import annotations

import functools
import http.server
import os
import threading
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8124
HOST = "127.0.0.1"


class PrototypeHandler(http.server.SimpleHTTPRequestHandler):
    """Serve mutable prototype assets without browser caching."""

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    os.chdir(HERE)
    handler = functools.partial(
        PrototypeHandler, directory=HERE
    )
    httpd = http.server.ThreadingHTTPServer((HOST, PORT), handler)
    url = f"http://{HOST}:{PORT}/?variant=B"
    if os.environ.get("THERMODENSE_FIGURE_REVIEW_NO_OPEN") != "1":
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    print(f"PROTOTYPE figure-review workbench -> {url}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
