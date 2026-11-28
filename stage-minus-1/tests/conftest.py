"""Test setup: put stage-minus-1 and stage-0a/research on the path."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT / "stage-0a" / "research"))
