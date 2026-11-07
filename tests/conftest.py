"""Test configuration — add Self-Distillation to import path for ssd package."""

import sys
from pathlib import Path

# Add Self-Distillation directory to path so the trainer packages (ssd, data, evals) import
SELF_DISTILLATION_DIR = Path(__file__).parent.parent / "src"
if str(SELF_DISTILLATION_DIR) not in sys.path:
    sys.path.insert(0, str(SELF_DISTILLATION_DIR))
