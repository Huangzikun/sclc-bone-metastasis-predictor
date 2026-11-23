"""特征筛选：单因素 -> 多因素 Logistic，p<0.05 入选"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import Config
from .io_utils import log, read_table, save_table

def _univariate_table(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for col in X.columns:
        try:
            Xc = sm.add_constant(X[[col]])
            res = sm.Logit(y, Xc).fit(disp=0)
            coef = res.params[col]
            orr = float(np.exp(coef))
            ci = res.conf_int().loc[col]
            lo, hi = float(np.exp(ci[0])), float(np.exp(ci[1]))
            p = float(res.pvalues[col])
        except Exception:
            orr, lo, hi, p = float("nan"), float("nan"), float("nan"), float("nan")
        rows.append({
            "自变量": col,
            "Odds Ratio": orr,
            "95% CI": f"({lo:.4f}, {hi:.4f})",
            "p-value": p,
            "p-display": "<0.001" if p < 0.001 else f"{p:.4f}",
        })
    return pd.DataFrame(rows)

def _multivariate_table(X_sel: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    Xm = sm.add_constant(X_sel)
    mres = sm.Logit(y, Xm).fit(disp=0)
    rows = []
    for var in X_sel.columns:
        coef = mres.params[var]
        orr = float(np.exp(coef))
        ci = mres.conf_int().loc[var]
        lo, hi = float(np.exp(ci[0])), float(np.exp(ci[1]))
        p = float(mres.pvalues[var])
        rows.append({
            "自变量": var,
            "Odds Ratio": orr,
            "95% CI": f"({lo:.4f}, {hi:.4f})",
            "p-value": p,
            "p-display": "<0.001" if p < 0.001 else f"{p:.4f}",
        })
    return pd.DataFrame(rows)

def run_select(cfg: Config) -> dict:
    fs = cfg.get("feature_selection", default={})
    target = fs.get("target", cfg.get("project", "target", default="Brain Mets"))
    uni_thr = fs.get("univariate_p_threshold", 0.05)
    multi_thr = fs.get("multivariate_p_threshold", 0.05)
    force = fs.get("force_include", []) or []

    X = read_table(cfg.artifacts_dir / "splits" / "X_train.csv")
    y = read_table(cfg.artifacts_dir / "splits" / "y_train.csv")[target]

    single = _univariate_table(X, y)
    out_dir = cfg.artifacts_dir / "selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_table(single, out_dir / "single_logit.csv")
    log("select", f"单因素逻辑回归完成，候选特征 {len(single)} 个。")

    selected = [r["自变量"] for _, r in single.iterrows() if r["p-value"] < uni_thr]
    for f in force:
        if f not in selected and f in X.columns:
            selected.append(f)
    log("select", f"单因素 p<{uni_thr} 入选 {len(selected)} 个: {selected}")

    X_sel = X[selected]
    multi = _multivariate_table(X_sel, y)
    save_table(multi, out_dir / "multi_logit.csv")
    multi_vars = [r["自变量"] for _, r in multi.iterrows() if r["p-value"] <= multi_thr]
    for f in force:
        if f not in multi_vars and f in selected:
            multi_vars.append(f)

    final = [c for c in selected if c in multi_vars]
    final = force + [c for c in final if c not in force]

    meta = {
        "target": target,
        "univariate_p_threshold": uni_thr,
        "multivariate_p_threshold": multi_thr,
        "force_include": force,
        "selected_features": final,
    }
    with open(out_dir / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log("select", f"最终筛选特征 {len(final)} 个: {final}")

    return {"single_logit": str(out_dir / "single_logit.csv"),
            "multi_logit": str(out_dir / "multi_logit.csv"),
            "selected_features": final}
