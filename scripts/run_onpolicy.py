#!/usr/bin/env python3
"""On-policy self-distillation: the student samples its own completions and is trained toward the teacher on every token.

    .venv/bin/python scripts/run_onpolicy.py configs/onpolicy_<task>.yaml --output runs/<name> [--seed 42]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch import main  # noqa: E402

if __name__ == "__main__":
    main("onpolicy")
