"""读写工具"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

def read_raw(path):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)

def read_table(path):
    path = Path(path)
    return pd.read_csv(path, index_col=0)

def save_table(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)

def save_series(se, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    se.to_csv(path)

def log(step, msg):
    print(f"[{step}] {msg}", flush=True)
