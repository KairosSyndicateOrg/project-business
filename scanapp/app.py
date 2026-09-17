"""
Entry point. Launches a pywebview window hosting gui/index.html, backed by
backend/api.py -> backend/pipeline.py (the §2 scan pipeline).

Run with: python app.py
"""

import os
import webview

from backend.api import Api

GUI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui", "index.html")


def main():
    api = Api()
    webview.create_window(
        "Scan Architecture — Prototype",
        GUI_PATH,
        js_api=api,
        width=1100,
        height=780,
        min_size=(800, 600),
    )
    webview.start()


if __name__ == "__main__":
    main()
