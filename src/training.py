"""模型训练：调参 -> 10折CV -> 校准 -> 阈值，结果写 artifacts/training/"""
from __future__ import annotations

import importlib
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from .config import Config
from .io_utils import log, read_table

try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

_SMOTE_KEYS = ("smote_on", "smote_strategy", "smote_k")

def _supports_class_weight(estimator):
    try:
        from inspect import signature
        return "class_weight" in signature(estimator.__init__).parameters
    except Exception:
        return False

def _classification_report_dict(y_true, y_pred, proba=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    f1 = f1_score(y_true, y_pred) if (tp + fp + fn) else 0.0
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, proba) if proba is not None else float("nan")
    ap = average_precision_score(y_true, proba) if proba is not None else float("nan")
    brier = brier_score_loss(y_true, proba) if proba is not None else float("nan")
    mcc = (matthews_corrcoef(y_true, y_pred)
           if len(np.unique(y_true)) > 1 else float("nan"))
    return {
        "sensitivity": round(sens, 4), "specificity": round(spec, 4),
        "ppv": round(ppv, 4), "npv": round(npv, 4),
        "f1": round(f1, 4), "accuracy": round(acc, 4),
        "auc": round(auc, 4) if proba is not None else float("nan"),
        "ap": round(ap, 4) if proba is not None else float("nan"),
        "brier": round(brier, 4) if proba is not None else float("nan"),
        "mcc": round(mcc, 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }

def _best_f1_threshold(y_true, proba_pos):
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_true, proba_pos)
    best_t, best_f1 = 0.5, -1.0
    for t, p, r in zip(thr, prec[:-1], rec[:-1]):
        if p + r == 0:
            continue
        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t

def _build_param_dist(spec):
    def _native(x):
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        return x

    if isinstance(spec, dict):
        if "arange" in spec:
            return [_native(v) for v in np.arange(*spec["arange"])]
        if "logspace" in spec:
            return [_native(v) for v in np.logspace(*spec["logspace"])]
        if "linspace" in spec:
            return [_native(v) for v in np.linspace(*spec["linspace"])]
        return _native(spec)
    if isinstance(spec, list):
        return [_native(v) for v in spec]
    return _native(spec)

def _instantiate(model_cfg: dict, params: dict):
    module = importlib.import_module(model_cfg["import"])
    cls = getattr(module, model_cfg["class"])
    return cls(**params)

def _make_smote(smote_cfg, y=None):
    if not smote_cfg.get("enabled", True):
        return "passthrough"
    k = int(smote_cfg.get("k_neighbors", 5))
    if y is not None:
        from collections import Counter
        counts = Counter(int(v) for v in np.asarray(y))
        n_minor, n_major = min(counts.values()), max(counts.values())
        natural_ratio = n_minor / n_major if n_major else 1.0
        req = float(smote_cfg.get("sampling_strategy", 0.3))
        if req < natural_ratio:
            return "passthrough"
        k = min(k, n_minor - 1) if n_minor > 1 else 1
    return SMOTE(sampling_strategy=float(smote_cfg.get("sampling_strategy", 0.3)),
                 random_state=int(smote_cfg.get("random_state", 42)),
                 k_neighbors=max(1, k))

def _make_scaler(scaler_name: str):
    if scaler_name == "standard":
        return StandardScaler()
    if scaler_name == "minmax":
        return MinMaxScaler()
    return "passthrough"

def _make_prep(onehot_cols, feats):
    cols = [c for c in (onehot_cols or []) if c in feats]
    if not cols:
        return "passthrough"
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cols)],
        remainder="passthrough", verbose_feature_names_out=True)

def build_pipeline(model_cfg, fixed, smote_cfg, scaler_name,
                   onehot_cols=None, global_class_weight=None, feats=None):
    fixed = dict(fixed or {})
    if global_class_weight:
        try:
            _mod = importlib.import_module(model_cfg["import"])
            _cls = getattr(_mod, model_cfg["class"])
            if _supports_class_weight(_cls):
                fixed = {**fixed, "class_weight": global_class_weight}
        except Exception:
            pass
    return Pipeline([
        ("prep", _make_prep(onehot_cols, feats or [])),
        ("scaler", _make_scaler(scaler_name)),
        ("smote", _make_smote(smote_cfg, None)),
        ("model", _instantiate(model_cfg, fixed)),
    ])

def _fit_pipeline(model_cfg, fixed, smote_cfg, scaler_name,
                  onehot_cols, global_class_weight, X, y):
    feats = list(X.columns) if hasattr(X, "columns") else []
    pipe = build_pipeline(model_cfg, fixed, smote_cfg, scaler_name,
                          onehot_cols, global_class_weight, feats)
    pipe.named_steps["smote"] = _make_smote(smote_cfg, y)
    pipe.fit(X, y)
    return pipe

def _suggest_param(spec, trial, key):
    vals = _build_param_dist(spec)
    if isinstance(vals, list):
        vals = [tuple(v) if isinstance(v, list) else v for v in vals]
        return trial.suggest_categorical(key, vals)
    return vals

def fit_best_model(mcfg, fixed, tr, X, y):
    """搜索最优超参，返回 (拟合好的 pipeline, best_params 记录)"""
    smote_cfg = tr.get("smote", {})
    scaler_name = tr.get("scaler", "standard")
    onehot_cols = tr.get("onehot_features") or []
    gcw = tr.get("class_weight", None)
    search_cfg = tr.get("search", {}) or {}
    srandom = int(search_cfg.get("random_state", 42))
    model_n_jobs = mcfg.get("search_n_jobs", search_cfg.get("n_jobs", -1))
    feats = list(X.columns) if hasattr(X, "columns") else []
    smote_on_cfg = bool(smote_cfg.get("enabled", True))

    def make(model_params=None, smote_step=None):
        pipe = build_pipeline(mcfg, {**fixed, **(model_params or {})}, smote_cfg,
                              scaler_name, onehot_cols, gcw, feats)
        if smote_step is not None:
            pipe.set_params(smote=smote_step)
        return pipe

    if not mcfg.get("search", True):
        pipe = _fit_pipeline(mcfg, fixed, smote_cfg, scaler_name,
                             onehot_cols, gcw, X, y)
        return pipe, {"search": False, "params": fixed}

    if search_cfg.get("method", "optuna") == "optuna" and _HAS_OPTUNA:
        n_trials = int(mcfg.get("search_n_trials", search_cfg.get("n_trials", 40)))
        n_cv = int(tr.get("cv", {}).get("folds", 5))
        scoring = tr.get("cv", {}).get("scoring", "roc_auc")

        def objective(trial):
            mp = {k: _suggest_param(spec, trial, k)
                  for k, spec in (mcfg.get("params") or {}).items()}
            if smote_on_cfg and trial.suggest_categorical("smote_on", [True, False]):
                pipe = make(mp)
                pipe.set_params(**{"smote__sampling_strategy": float(
                    trial.suggest_categorical("smote_strategy", [0.4, 0.5, 0.6, 0.8, 1.0]))})
                pipe.set_params(**{"smote__k_neighbors": int(
                    trial.suggest_categorical("smote_k", [3, 5, 10]))})
            elif smote_on_cfg:
                pipe = make(mp, smote_step="passthrough")
            else:
                pipe = make(mp)
            scores = cross_val_score(pipe, X, y,
                                     cv=StratifiedKFold(n_splits=n_cv, shuffle=True,
                                                        random_state=srandom),
                                     scoring=scoring, n_jobs=model_n_jobs)
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize",
                                    sampler=TPESampler(seed=srandom))
        study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

        best = study.best_params
        mp = {k: v for k, v in best.items() if k not in _SMOTE_KEYS}
        smote_used = bool(smote_on_cfg and best.get("smote_on", True))
        if smote_used:
            pipe = make(mp)
            pipe.set_params(**{"smote__sampling_strategy": float(best.get("smote_strategy",
                                smote_cfg.get("sampling_strategy", 0.5)))})
            pipe.set_params(**{"smote__k_neighbors": int(best.get("smote_k",
                                smote_cfg.get("k_neighbors", 5)))})
        else:
            pipe = make(mp, smote_step="passthrough" if smote_on_cfg else None)
        pipe.fit(X, y)
        return pipe, {
            "search": "optuna",
            "best_params": _to_native(mp),
            "smote": {"enabled": smote_used,
                      "sampling_strategy": (float(best.get("smote_strategy",
                                                smote_cfg.get("sampling_strategy", 0.5)))
                                            if smote_used else None),
                      "k_neighbors": (int(best.get("smote_k",
                                        smote_cfg.get("k_neighbors", 5)))
                                      if smote_used else None)},
            "cv_auc": round(float(study.best_value), 4),
        }

    dist = {f"model__{k}": _build_param_dist(v) for k, v in (mcfg.get("params") or {}).items()}
    try:
        rsearch = RandomizedSearchCV(
            estimator=build_pipeline(mcfg, fixed, smote_cfg, scaler_name,
                                     onehot_cols, gcw, feats),
            param_distributions=dist,
            n_iter=int(search_cfg.get("n_iter", 50)),
            cv=int(tr.get("cv", {}).get("folds", 5)),
            scoring=tr.get("cv", {}).get("scoring", "roc_auc"),
            n_jobs=model_n_jobs, random_state=srandom)
        rsearch.fit(X, y)
        pipe = rsearch.best_estimator_
        return pipe, {"search": "random",
                      "best_params": _to_native({k.replace("model__", ""): v
                                                 for k, v in rsearch.best_params_.items()})}
    except Exception as e:
        log("train", f"  {mcfg['name']} 搜索失败({e})，回退默认参数。")
        pipe = _fit_pipeline(mcfg, fixed, smote_cfg, scaler_name, onehot_cols, gcw, X, y)
        return pipe, {"search": False, "params": fixed, "error": str(e)}

def calibrate_model(pipe, tr, X, y):
    cc = tr.get("calibration", {}) or {}
    if not cc.get("enabled", True):
        return pipe, None
    cal = CalibratedClassifierCV(estimator=pipe, method=cc.get("method", "isotonic"),
                                 cv=int(cc.get("folds", 5)))
    cal.fit(X, y)
    return cal, {"method": cc.get("method", "isotonic"), "folds": int(cc.get("folds", 5))}

def _to_native(obj):
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def _resolve_features(cfg: Config) -> list[str]:
    fs = cfg.get("feature_selection", default={})
    if cfg.get("training", "features_source", default="selected") == "explicit":
        feats = fs.get("explicit_features", [])
        log("train", f"使用配置显式特征({len(feats)}): {feats}")
        return feats
    meta_path = cfg.artifacts_dir / "selection" / "selected_features.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        feats = meta.get("selected_features", [])
        log("train", f"使用 select 阶段筛选特征({len(feats)}): {feats}")
        return feats
    feats = fs.get("explicit_features", [])
    log("train", "未找到 selected 结果，回退到显式特征列表。")
    return feats

def run_train(cfg):
    tr = cfg.get("training", default={})
    target = tr.get("target", cfg.get("project", "target", default="Brain Mets"))
    feats = _resolve_features(cfg)
    if not feats:
        raise RuntimeError("未解析到任何特征，请先运行 select 阶段或配置 explicit_features。")

    sp = cfg.artifacts_dir / "splits"
    X_train = read_table(sp / "X_train.csv")[feats]
    y_train = read_table(sp / "y_train.csv")[target]
    X_val = read_table(sp / "X_val.csv")[feats]
    y_val = read_table(sp / "y_val.csv")[target]
    X_ext = read_table(sp / "X_external.csv")[feats]
    y_ext = read_table(sp / "y_external.csv")[target]
    log("train", f"训练集 {X_train.shape}，内部验证 {X_val.shape}，外部验证 {X_ext.shape}")

    search_cfg = tr.get("search", {}) or {}
    srandom = int(search_cfg.get("random_state", 42))
    compare_folds = int(tr.get("cv", {}).get("compare_folds", 10))
    scoring = tr.get("cv", {}).get("scoring", "roc_auc")
    global_n_jobs = search_cfg.get("n_jobs", -1)
    method_desc = search_cfg.get("method", "optuna") if _HAS_OPTUNA else "random"

    out_dir = cfg.artifacts_dir / "training"
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    cv_scores, metrics_val, metrics_ext, metrics_ext_raw = {}, {}, {}, {}
    predictions_val, predictions_ext, proba_ext_df = {}, {}, {}
    best_thresholds, best_params = {}, {}

    for mcfg in tr.get("models", []):
        name = mcfg["name"]
        log("train", f"训练模型 {name} ...")
        fixed = {k: v for k, v in (mcfg.get("fixed_params") or {}).items()}
        model_n_jobs = mcfg.get("search_n_jobs", global_n_jobs)

        raw, record = fit_best_model(mcfg, fixed, tr, X_train, y_train)
        best_params[name] = record
        log("train", f"  {name} 搜索完成({record['search']}): {record.get('best_params', fixed)}"
                     + (f" cv_auc={record['cv_auc']}" if record.get("cv_auc") else ""))

        skf = StratifiedKFold(n_splits=compare_folds, shuffle=True, random_state=srandom)
        scores = cross_val_score(raw, X_train, y_train, cv=skf, scoring=scoring,
                                 n_jobs=model_n_jobs)
        cv_scores[name] = scores

        cal, cal_info = calibrate_model(raw, tr, X_train, y_train)
        if cal_info:
            log("train", f"  {name} 概率校准: {cal_info['method']} × {cal_info['folds']} 折")

        joblib.dump(cal, models_dir / f"{name}.joblib")
        proba_val = cal.predict_proba(X_val)[:, 1]
        proba_ext = cal.predict_proba(X_ext)[:, 1]
        thr = _best_f1_threshold(y_val, proba_val)
        best_thresholds[name] = round(float(thr), 4)

        pred_val_t = (proba_val >= thr).astype(int)
        pred_ext_t = (proba_ext >= thr).astype(int)
        pred_ext_raw = (proba_ext >= 0.5).astype(int)

        metrics_val[name] = _classification_report_dict(y_val, pred_val_t, proba_val)
        metrics_ext[name] = _classification_report_dict(y_ext, pred_ext_t, proba_ext)
        metrics_ext_raw[name] = _classification_report_dict(y_ext, pred_ext_raw, proba_ext)
        predictions_val[name] = pred_val_t
        predictions_ext[name] = pred_ext_t
        proba_ext_df[name] = proba_ext
        log("train", f"  {name} 阈值={thr:.3f} | 外部 F1={metrics_ext[name]['f1']:.3f} "
                     f"灵敏度={metrics_ext[name]['sensitivity']:.3f} "
                     f"AUC={metrics_ext[name]['auc']:.3f} | cv_auc={scores.mean():.3f}")

    pd.DataFrame(cv_scores).to_csv(out_dir / "cv_scores.csv", index=False)
    pd.DataFrame.from_dict(metrics_val, orient="index").reset_index().rename(
        columns={"index": "model"}).to_csv(out_dir / "metrics_val.csv", index=False)
    pd.DataFrame.from_dict(metrics_ext, orient="index").reset_index().rename(
        columns={"index": "model"}).to_csv(out_dir / "metrics_external.csv", index=False)
    pd.DataFrame.from_dict(metrics_ext_raw, orient="index").reset_index().rename(
        columns={"index": "model"}).to_csv(out_dir / "metrics_external_default05.csv", index=False)
    pd.DataFrame(predictions_val).to_csv(out_dir / "predictions_val.csv", index=False)
    pd.DataFrame(predictions_ext).to_csv(out_dir / "predictions_external.csv", index=False)
    pd.DataFrame(proba_ext_df).to_csv(out_dir / "predictions_external_proba.csv", index=False)
    with open(out_dir / "best_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(best_thresholds, f, ensure_ascii=False, indent=2)
    with open(out_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(_to_native(best_params), f, ensure_ascii=False, indent=2)

    log("train", f"训练完成（搜索口径={method_desc}），结果已保存到 {out_dir}")
    return {"cv_scores": str(out_dir / "cv_scores.csv"),
            "metrics_val": str(out_dir / "metrics_val.csv"),
            "metrics_external": str(out_dir / "metrics_external.csv"),
            "best_params": str(out_dir / "best_params.json"),
            "best_thresholds": str(out_dir / "best_thresholds.json")}
