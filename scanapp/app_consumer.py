"""
Entry point for the consumer-facing shop app (Analytics / Inventory / AI
Chat / Options). Shares the exact same backend/api.py and database as
app.py (the scan-architecture debug shell) — run either one against the
same scan_app.db.

Run with: python app_consumer.py
"""

import os
import webview

from backend.api import Api

GUI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui", "consumer.html")


def main():
    api = Api()
    webview.create_window(
        "Kairos",
        GUI_PATH,
        js_api=api,
        width=430,
        height=860,
        min_size=(360, 640),
    )
    webview.start()


if __name__ == "__main__":
    main()
