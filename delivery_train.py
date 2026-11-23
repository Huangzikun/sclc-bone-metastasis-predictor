"""交付训练：跑全部模型，把预测/指标/CV/SHAP 缓存给 delivery_render.py 用"""
import sys, json, re, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance
from catboost import Pool

warnings.filterwarnings("ignore")

ROOT = Path(".")
ARTS = ROOT / "artifacts"
SPLITS = ARTS / "splits"
DELIV = ROOT / "delivery"
CACHE = ARTS / "delivery_cache"
CACHE.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT.resolve()))
from src.training import (
    _classification_report_dict, _best_f1_threshold, fit_best_model, calibrate_model,
)

import yaml
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
target = cfg["project"]["target"]
tr = cfg["training"]

if tr.get("features_source", "selected") == "selected":
    feats = json.load(open(ARTS / "selection" / "selected_features.json",
                           encoding="utf-8"))["selected_features"]
else:
    feats = list(cfg["feature_selection"]["explicit_features"])

COMPARE_FOLDS = int(tr.get("cv", {}).get("compare_folds", 10))
srandom = int(tr.get("search", {}).get("random_state", 42))
global_n_jobs = tr.get("search", {}).get("n_jobs", -1)
models_cfg = tr.get("models", [])

def load(p):
    d = pd.read_csv(p)
    if "Unnamed: 0" in d.columns:
        d = d.drop(columns=["Unnamed: 0"])
    return d

Xtr = load(SPLITS / "X_train.csv").drop(columns=["Race"], errors="ignore")
ytr = load(SPLITS / "y_train.csv").iloc[:, 0]
Xval = load(SPLITS / "X_val.csv").drop(columns=["Race"], errors="ignore")
yval = load(SPLITS / "y_val.csv").iloc[:, 0]
Xex = load(SPLITS / "X_external.csv").drop(columns=["Race"], errors="ignore")
yex = load(SPLITS / "y_external.csv").iloc[:, 0]
Xtr_f, Xval_f, Xex_f = Xtr[feats], Xval[feats], Xex[feats]
print(f"[split] 训练集 = 纯内部 X_train {len(Xtr_f)} 例（10 折 CV 在其内部进行）；"
      f"内部验证 X_val {len(Xval_f)} 例；外部验证 {len(Xex_f)} 例")

split_seed = int(cfg.get("splitting", {}).get("random_state", 42))

cache = {
    "feats": feats, "target": target,
    "n_train": int(len(Xtr_f)), "n_val": int(len(Xval_f)), "n_external": int(len(Xex_f)),
    "split_seed": split_seed,
    "y_train": ytr.values.tolist(),
    "y_val": yval.values.tolist(),
    "y_external": yex.values.tolist(),
}

cv_scores = {}; metrics = {}; proba = {}; best_params = {}; best_th = {}
trained = {}
trained_raw = {}

for mcfg in models_cfg:
    name = mcfg["name"]
    print(f"[train] {name} ...")
    fixed = {k: v for k, v in (mcfg.get("fixed_params") or {}).items()}
    if name == "MLP":
        fixed = {**fixed, "max_iter": 300}
    model_n_jobs = mcfg.get("search_n_jobs", global_n_jobs)

    raw, record = fit_best_model(mcfg, fixed, tr, Xtr_f, ytr)
    best_params[name] = record
    trained_raw[name] = raw
    print(f"  {name} 搜索({record['search']}): {record.get('best_params', fixed)}"
          + (f" | 内部CV AUC={record['cv_auc']}" if record.get("cv_auc") else ""))

    skf = StratifiedKFold(n_splits=COMPARE_FOLDS, shuffle=True, random_state=srandom)
    cv = cross_val_score(raw, Xtr_f, ytr, cv=skf, scoring="roc_auc", n_jobs=model_n_jobs)
    cv_scores[name] = cv.tolist()

    cal, cal_info = calibrate_model(raw, tr, Xtr_f, ytr)
    trained[name] = cal
    if cal_info:
        print(f"  {name} 概率校准: {cal_info['method']} x {cal_info['folds']} 折")

    p_tr = cal.predict_proba(Xtr_f)[:, 1]
    p_val = cal.predict_proba(Xval_f)[:, 1]
    p_ex = cal.predict_proba(Xex_f)[:, 1]
    proba[name] = {"train": p_tr.tolist(), "val": p_val.tolist(), "ext": p_ex.tolist()}

    thr = _best_f1_threshold(yval.values, p_val)
    best_th[name] = float(thr)
    hard = lambda p: (np.asarray(p) >= thr).astype(int)
    metrics[name] = {
        "train": _classification_report_dict(ytr.values, hard(p_tr), p_tr),
        "val": _classification_report_dict(yval.values, hard(p_val), p_val),
        "ext": _classification_report_dict(yex.values, hard(p_ex), p_ex),
    }
    print(f"  {name} 阈值={thr:.3f} | ext AUC={metrics[name]['ext']['auc']:.3f} "
          f"F1={metrics[name]['ext']['f1']:.3f} | cv_auc={cv.mean():.3f}")

perm_imp_ext = {}
for name, model in trained.items():
    try:
        pi = permutation_importance(model, Xex_f, yex.values, n_repeats=5,
                                    random_state=42, n_jobs=global_n_jobs, scoring="roc_auc")
        perm_imp_ext[name] = pi.importances_mean.tolist()
    except Exception as e:
        print(f"  {name} 排列重要性失败({e})")
        perm_imp_ext[name] = [float("nan")] * len(feats)
print("[perm_imp_ext] done")

cache.update({
    "cv_scores": cv_scores, "metrics": metrics, "proba": proba,
    "best_params": best_params, "best_th": best_th,
    "perm_imp_ext": perm_imp_ext,
    "shap_summary_ext": None, "shap_feat_ext": None, "shap_names_ext": None,
})
with open(CACHE / "delivery_cache.pkl", "wb") as f:
    pickle.dump(cache, f)
print(f"[done] 主缓存已保存到 {CACHE / 'delivery_cache.pkl'}")

shap_raw = trained_raw.get("CatBoost")
try:
    Xt, names_t = Xex_f, list(feats)
    for step_name in ("prep", "scaler"):
        step = shap_raw.named_steps.get(step_name)
        if step is not None and step != "passthrough":
            Xt = step.transform(Xt)
            if step_name == "prep":
                names_t = [re.sub(r"^(num|cat|remainder)__", "", n)
                           for n in step.get_feature_names_out()]
    contribs = np.asarray(shap_raw.named_steps["model"].get_feature_importance(
        Pool(Xt), type="ShapValues"))[:, :-1]
    col_groups = {f: [] for f in feats}
    for j, name in enumerate(names_t):
        base = name.rsplit("_", 1)[0]
        if base not in col_groups:
            raise KeyError(f"变换列 {name!r} 无法映射回原始特征")
        col_groups[base].append(j)
    agg = np.column_stack([contribs[:, col_groups[f]].sum(axis=1) for f in feats])
    cache["shap_summary_ext"] = agg.tolist()
    cache["shap_feat_ext"] = Xex_f.to_numpy(dtype=float).tolist()
    cache["shap_names_ext"] = list(feats)
    print(f"[shap] CatBoost ShapValues aggregated to {len(feats)} original features: {agg.shape}")
    with open(CACHE / "delivery_cache.pkl", "wb") as f:
        pickle.dump(cache, f)
except Exception as e:
    print(f"[shap] 失败({e})，跳过 SHAP（不影响其余交付）")

print(f"[summary] ext AUC: " + ", ".join(f"{n}={metrics[n]['ext']['auc']:.3f}" for n in metrics))
print(f"[summary] 阈值来源 = 内部验证集(X_val) max-F1；外部验证集未参与任何调优")
