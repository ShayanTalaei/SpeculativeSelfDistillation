#!/usr/bin/env python3
"""Speculative self-distillation: the student drafts, the teacher takes over on high-divergence tokens, two GPUs.

    .venv/bin/python scripts/run_ssd.py configs/ssd_<task>.yaml --output runs/<name> [--seed 42]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch import main  # noqa: E402

if __name__ == "__main__":
    main("ssd")
