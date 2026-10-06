"""Small torch-free helpers shared by the phase modules.

Kept apart from ``encscore`` (which imports torch at module level) so that
numpy-only phases such as ``augdetect`` can be imported and tested without
the neural extra. ``encscore`` re-exports both names for older importers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _abs(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p
