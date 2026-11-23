"""数据清洗：类别编码 + 列名映射，规则都在 config.cleaning"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

from .config import Config
from .io_utils import log, read_raw, save_table

def _encode_column(series, spec, unknown_code):
    if spec.get("type") == "age":
        def f(x):
            s = str(x).strip()
            if s == "90+":
                return 90
            try:
                return int(float(s))
            except ValueError:
                return unknown_code
        return series.apply(f)

    mapping = spec.get("map", {})
    norm_map = {str(k).strip(): v for k, v in mapping.items()}
    default = spec.get("default", unknown_code)

    def f(x):
        return norm_map.get(str(x).strip(), default)
    return series.apply(f)

def clean_dataframe(df, cfg):
    clean_cfg = cfg.get("cleaning", default={})
    rename = clean_cfg.get("rename", {})
    unknown_code = clean_cfg.get("unknown_code", 7)

    df = df.rename(columns={k: v for k, v in rename.items()})
    df.columns = [c.strip() for c in df.columns]

    drop_cols = [c for c in (clean_cfg.get("drop_columns", []) or [])
                 if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        log("clean", f"已删除配置指定的列: {drop_cols}")

    if clean_cfg.get("already_encoded", False):
        df = df.apply(
            lambda s: pd.to_numeric(
                s.astype(str).str.replace("+", "", regex=False).str.strip(),
                errors="coerce",
            )
        )
        return df.fillna(unknown_code)

    encodings = clean_cfg.get("encodings", {})
    for col, spec in encodings.items():
        if col in df.columns:
            df[col] = _encode_column(df[col], spec, unknown_code)
        else:
            log("clean", f"警告: 编码配置中存在列 '{col}'，但数据中未找到，已跳过。")
    return df

def run_clean(cfg):
    artifacts = cfg.artifacts_dir
    out_dir = artifacts / "cleaned"
    out_dir.mkdir(parents=True, exist_ok=True)

    internal_raw = cfg.resolve(cfg.get("paths", "raw_internal"))
    external_raw = cfg.resolve(cfg.get("paths", "raw_external"))
    external_supp = cfg.resolve(cfg.get("paths", "raw_external_supplement"))

    log("clean", f"读取内部队列: {internal_raw}")
    in_df = read_raw(internal_raw)
    log("clean", f"内部队列形状: {in_df.shape}")

    log("clean", f"读取外部队列: {external_raw}")
    out_df = read_raw(external_raw)
    if external_supp and Path(external_supp).exists():
        log("clean", f"读取外部补充: {external_supp}")
        out_supp = read_raw(external_supp)
        out_df = pd.concat([out_df, out_supp], ignore_index=True)
    log("clean", f"外部队列形状: {out_df.shape}")

    in_clean = clean_dataframe(in_df, cfg)
    out_clean = clean_dataframe(out_df, cfg)

    in_path = out_dir / "cleaned_internal.csv"
    out_path = out_dir / "cleaned_external.csv"
    save_table(in_clean, in_path)
    save_table(out_clean, out_path)
    log("clean", f"已保存清洗结果 -> {in_path.name}, {out_path.name}")

    return {"cleaned_internal": in_path, "cleaned_external": out_path}
