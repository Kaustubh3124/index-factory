from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml


def _ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    return obj


def load_config(path: str | Path) -> SimpleNamespace:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    cfg = _ns(raw)
    cfg.base_date = pd.Timestamp(cfg.base_date)
    cfg._path = str(path)
    return cfg
