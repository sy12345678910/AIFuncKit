from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src", ROOT / "src" / "model_provide"):
    path_str = str(entry)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
