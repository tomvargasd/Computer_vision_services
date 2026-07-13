#!/usr/bin/env python3
"""
CVVision v2.0 — Computer Vision Dashboard
Entry point. Run with: python3 app.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.app import app

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001, use_reloader=False, threaded=True)
