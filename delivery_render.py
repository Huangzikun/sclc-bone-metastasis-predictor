"""渲染交付图表：读 delivery_cache.pkl，出 PNG 图 + XLSX 表到 delivery/"""
import warnings, json, pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
import openpyxl
from openpyxl.utils import get_column_letter
from scipy import stats as sps
import statsmodels.api as sm
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, brier_score_loss, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
)
from sklearn.calibration import calibration_curve
warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(".")
ARTS = ROOT / "artifacts"
SPLITS = ARTS / "splits"
DELIV = ROOT / "delivery"
CACHE = ARTS / "delivery_cache"
DELIV.mkdir(parents=True, exist_ok=True)

import yaml
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
target = cfg["project"]["target"]
if cfg["training"].get("features_source", "selected") == "selected":
    feats = json.load(open(ARTS / "selection" / "selected_features.json",
                           encoding="utf-8"))["selected_features"]
else:
    feats = list(cfg["feature_selection"]["explicit_features"])
TARGET_LABEL = "Bone metastasis (+)"

ORDER = ["XGB", "RF", "DT", "KNN", "MLP", "SVM", "LR", "BNB", "LGBM", "CatBoost"]

import re
def _safe(f): return re.sub(r"\W+", "_", f)
SAFE = {f: _safe(f) for f in feats}
RSAFE = {v: k for k, v in SAFE.items()}

C = pickle.load(open(CACHE / "delivery_cache.pkl", "rb"))
metrics = C["metrics"]; proba = C["proba"]; cv = C["cv_scores"]
perm_imp_ext = C.get("perm_imp_ext")
shap_sum_ext = C.get("shap_summary_ext"); shap_feat_ext = C.get("shap_feat_ext")
shap_names_ext = C.get("shap_names_ext")

def load(p):
    d = pd.read_csv(p)
    if "Unnamed: 0" in d.columns: d = d.drop(columns=["Unnamed: 0"])
    return d
Xtr = load(SPLITS / "X_train.csv").drop(columns=["Race"], errors="ignore")
ytr = load(SPLITS / "y_train.csv").iloc[:, 0]
Xval = load(SPLITS / "X_val.csv").drop(columns=["Race"], errors="ignore")
yval = load(SPLITS / "y_val.csv").iloc[:, 0]
Xex = load(SPLITS / "X_external.csv").drop(columns=["Race"], errors="ignore")
yex = load(SPLITS / "y_external.csv").iloc[:, 0]
Xtr_f, Xval_f, Xex_f = Xtr[feats], Xval[feats], Xex[feats]
Xint = pd.concat([Xtr_f, Xval_f], axis=0, ignore_index=True)
yint = pd.concat([ytr, yval], axis=0, ignore_index=True)
N_INT = len(Xint); N_TR = len(Xtr_f); N_VAL = len(Xval_f); N_EX = len(Xex_f)

CONTINUOUS = {"Age", "Tumor Size (mm)"}

LABELS = {
    "Gender": {0: "Male", 1: "Female"},
    "T stage": {0: "T1", 1: "T2", 2: "T3", 3: "T4"},
    "N stage": {0: "N0", 1: "N1", 2: "N2", 3: "N3"},
    "Brain Mets": {0: "No", 1: "Yes"},
    "Mets at DX-liver": {0: "No", 1: "Yes", 2: "Unknown"},
    "Mets at DX-lung": {0: "No", 1: "Yes", 2: "Unknown"},
    "Mets at DX-Distant LN": {0: "No", 1: "Yes", 2: "Unknown"},
    "Marital status": {0: "Divorced", 1: "Married", 2: "Separated", 3: "Single",
                       4: "Unmarried/Domestic", 5: "Widowed", 6: "Unknown"},
    "Primary site": {}, "Laterality": {},
    "Separate Tumor Nodules Ipsilateral Lung": {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"},
}
def lvl(f, v):
    if v in (2, 7): return "Unknown"
    return str(LABELS.get(f, {}).get(v, v))

def save(fig, name):
    p = DELIV / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] 已保存 {name}")

def dca_net_benefit(y_true, p, pts):
    y_true = np.asarray(y_true); p = np.asarray(p)
    nb = []
    for pt in pts:
        yhat = (p >= pt).astype(int)
        tp = ((yhat == 1) & (y_true == 1)).sum()
        fp = ((yhat == 1) & (y_true == 0)).sum()
        nb.append((tp - fp * (pt / (1 - pt))) / len(y_true))
    return np.array(nb)

def fig2():
    corr = Xint[feats].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats))); ax.set_yticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(feats, fontsize=7)
    for i in range(len(feats)):
        for j in range(len(feats)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Figure 2. Spearman correlation matrix of %d variables\n(Original internal cohort, n=%d)" % (len(feats), N_INT), fontsize=11)
    save(fig, "Figure 2. Spearman correlation heatmap.png")

def fig3():
    """
    按 Word 投稿文件 Figure 3 参考格式绘制：
      - 标题：10-Fold Cross-Validation Performance Comparison
      - 折线+圆点，x=10 折，每条线一个模型
      - 浅色网格、无顶部/右侧厚重边框
      - 图例置于图内右下，2 列紧凑布局，落在 0.4–0.58 的空白区，不遮挡曲线
      - 图例格式 "Model: AUC Mean=..., Std=..."
      - 图例顺序对齐 Word：SVM, KNN, MLP, DT, BNB, LR, RF, XGB
    """
    colors = {
        "SVM": "#E74C3C", "KNN": "#17BECF", "MLP": "#85C1E9", "DT": "#82E0AA",
        "BNB": "#F4D03F", "LR": "#A9DFBF", "RF": "#C0CA33", "XGB": "#AF7AC5",
        "LGBM": "#393B79", "CatBoost": "#637939",
    }
    legend_order = ["SVM", "KNN", "MLP", "DT", "BNB", "LR", "RF", "XGB", "LGBM", "CatBoost"]
    folds = list(range(1, len(cv[ORDER[0]]) + 1))

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for m in ORDER:
        vals = cv[m]
        mu, sd = float(np.mean(vals)), float(np.std(vals))
        ax.plot(folds, vals, marker="o", ms=4, lw=1.6, color=colors.get(m),
                label=f"{m}: AUC Mean={mu:.4f}, Std={sd:.4f}")

    ax.set_xticks(folds)
    ax.set_xlabel("Cross-Validation Folds", fontsize=11)
    ax.set_ylabel("AUC Score", fontsize=11)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlim(0.8, len(folds) + 0.2)
    ax.grid(True, ls="--", lw=0.5, alpha=0.35)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_linewidth(0.8)
        ax.spines[sp].set_color("#888888")

    handles, labels = ax.get_legend_handles_labels()
    label2handle = dict(zip(labels, handles))
    ordered_labels = [
        f"{m}: AUC Mean={float(np.mean(cv[m])):.4f}, Std={float(np.std(cv[m])):.4f}"
        for m in legend_order
    ]
    ordered_handles = [label2handle[lbl] for lbl in ordered_labels]
    ax.legend(ordered_handles, ordered_labels, loc="lower right", fontsize=8,
              ncol=2, columnspacing=1.0, handlelength=2.0,
              framealpha=0.95, edgecolor="#AAAAAA", facecolor="white",
              fancybox=True, shadow=False)

    ax.set_title("10-Fold Cross-Validation Performance Comparison",
                 fontsize=13, fontweight="bold", pad=12)
    save(fig, "Figure 3. AUC of 10-fold cross-validation.png")

def fig4():
    """
    按 Word 文档 Figure 4 参考格式，把 ROC/PR/DCA/校准 拆成 2 组 × 4 面板 = 8 张独立单图，
    存入 delivery/Figure 4. Training set ROC PR calibration DCA/ 子文件夹：
      A-D：内部训练集 X_train（模型在 X_train 上训练并在其上预测，in-sample，n=6123）
      E-H：内部验证集 X_val（模型在 X_train 训练后预测 X_val，n=2625）
    样式：标题、浅色网格、无顶部/右侧厚重边框、图例置于右侧外部（不遮挡曲线）。
    """
    sub = DELIV / "Figure 4. Training set ROC PR calibration DCA"
    sub.mkdir(parents=True, exist_ok=True)
    colors = {
        "SVM": "#E74C3C", "KNN": "#17BECF", "MLP": "#85C1E9", "DT": "#82E0AA",
        "BNB": "#F4D03F", "LR": "#A9DFBF", "RF": "#C0CA33", "XGB": "#AF7AC5",
        "LGBM": "#393B79", "CatBoost": "#637939",
    }
    groups = {
        "train": {
            "y": np.asarray(C["y_train"]),
            "p": {m: np.asarray(C["proba"][m]["train"]) for m in ORDER},
            "n": int(len(np.asarray(C["y_train"]))),
            "label": "training set",
        },
        "ival": {
            "y": np.asarray(C["y_val"]),
            "p": {m: np.asarray(C["proba"][m]["val"]) for m in ORDER},
            "n": int(len(np.asarray(C["y_val"]))),
            "label": "internal validation set",
        },
    }
    n_tr, n_iv = groups["train"]["n"], groups["ival"]["n"]
    panels = [
        ("roc", f"A. ROC curves of different ML models in the training set (n={n_tr})",
         "False positive rate", "True positive rate",
         "A. ROC curves of training set.png", "train"),
        ("pr",  f"B. PR curves of different ML models in the training set (n={n_tr})",
         "Recall", "Precision",
         "B. PR curves of training set.png", "train"),
        ("dca", f"C. DCA curves of different ML models in the training set (n={n_tr})",
         "Threshold probability", "Net benefit",
         "C. DCA curves of training set.png", "train"),
        ("calibration", f"D. Calibration curves of different ML models in the training set (n={n_tr})",
         "Predicted probability", "Actual probability",
         "D. Calibration curves of training set.png", "train"),
        ("roc", f"E. ROC curves of different ML models in the internal validation set (n={n_iv})",
         "False positive rate", "True positive rate",
         "E. ROC curves of validation set.png", "ival"),
        ("pr",  f"F. PR curves of different ML models in the internal validation set (n={n_iv})",
         "Recall", "Precision",
         "F. PR curves of validation set.png", "ival"),
        ("dca", f"G. DCA curves of different ML models in the internal validation set (n={n_iv})",
         "Threshold probability", "Net benefit",
         "G. DCA curves of validation set.png", "ival"),
        ("calibration", f"H. Calibration curves of different ML models in the internal validation set (n={n_iv})",
         "Predicted probability", "Actual probability",
         "H. Calibration curves of validation set.png", "ival"),
    ]
    for kind, title, xl, yl, fname, gk in panels:
        g = groups[gk]
        y = g["y"]; pdict = g["p"]
        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        if kind == "roc":
            for m in ORDER:
                fpr, tpr, _ = roc_curve(y, pdict[m])
                ax.plot(fpr, tpr, lw=1.7, color=colors[m],
                        label=f"{m} (AUC={auc(fpr, tpr):.3f})")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        elif kind == "pr":
            for m in ORDER:
                prec, rec, _ = precision_recall_curve(y, pdict[m])
                ax.plot(rec, prec, lw=1.7, color=colors[m],
                        label=f"{m} (AP={auc(rec, prec):.4f})")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        elif kind == "dca":
            pts = np.linspace(0.01, 0.99, 50)
            prev = y.mean()
            ax.plot(pts, prev - (1 - prev) * pts / (1 - pts), "k--", lw=1, label="Assume all")
            ax.plot(pts, [0] * len(pts), "k:", lw=1, label="Assume none")
            for m in ORDER:
                nb = dca_net_benefit(y, pdict[m], pts)
                ax.plot(pts, nb, lw=1.5, color=colors[m], label=m)
        else:
            for m in ORDER:
                frac_pos, mean_pred = calibration_curve(
                    y, pdict[m], n_bins=10, strategy="quantile")
                ax.plot(mean_pred, frac_pos, "s-", lw=1.4, ms=5, color=colors[m], label=m)
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel(xl, fontsize=11)
        ax.set_ylabel(yl, fontsize=11)
        ax.grid(True, ls="--", lw=0.5, alpha=0.35)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]:
            ax.spines[sp].set_linewidth(0.8)
            ax.spines[sp].set_color("#888888")
        ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  framealpha=0.95, edgecolor="#AAAAAA", facecolor="white",
                  fancybox=True, shadow=False)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
        p = sub / fname
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig] 已保存 {p.name}")

def fig5():
    """
    参考 Word 文档 Figure 5 格式：8 模型 × 8 指标的热力图，数值着色 0–1，右侧颜色条。
    生成三张独立图片，分别对应训练集、内部验证集、外部验证集：
      A. training set
      B. internal validation set
      C. external validation set
    存放于 delivery/Figure 5. Model performance train validation/。
    """
    sub = DELIV / "Figure 5. Model performance train validation"
    sub.mkdir(parents=True, exist_ok=True)

    old_png = DELIV / "Figure 5. Model performance train validation.png"
    if old_png.exists():
        old_png.unlink()
        print(f"[fig] 移除旧版合并图 {old_png.name}")

    FIG5_ORDER = ["SVM", "KNN", "MLP", "DT", "BNB", "LR", "RF", "XGB", "LGBM", "CatBoost"]
    FIG5_METRICS = ["AUC", "Accuracy", "Precision", "Recall",
                    "F1-Measure", "MCC", "Specificity", "Brier Score"]

    def _perf_matrix(y_true, pdict):
        """8 模型 × 8 指标矩阵；阈值统一用缓存中冻结的（内部验证集 max-F1）阈值，
        与 Table 4/S3/S4 口径一致。"""
        y_true = np.asarray(y_true)
        mat = np.zeros((len(FIG5_ORDER), len(FIG5_METRICS)), dtype=float)
        for i, m in enumerate(FIG5_ORDER):
            p = np.asarray(pdict[m])
            pred = (p >= C["best_th"][m]).astype(int)
            mat[i, 0] = roc_auc_score(y_true, p)
            mat[i, 1] = accuracy_score(y_true, pred)
            mat[i, 2] = precision_score(y_true, pred, zero_division=0)
            mat[i, 3] = recall_score(y_true, pred, zero_division=0)
            mat[i, 4] = f1_score(y_true, pred, zero_division=0)
            mat[i, 5] = matthews_corrcoef(y_true, pred)
            tn = int(((pred == 0) & (y_true == 0)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            mat[i, 6] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            mat[i, 7] = brier_score_loss(y_true, p)
        return mat

    def _draw(mat, title, fname, n):
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")

        ax.set_xticks(np.arange(len(FIG5_METRICS)))
        ax.set_yticks(np.arange(len(FIG5_ORDER)))
        ax.set_xticklabels(FIG5_METRICS, fontsize=10)
        ax.set_yticklabels(FIG5_ORDER, fontsize=10)

        ax.set_xticks(np.arange(len(FIG5_METRICS)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(FIG5_ORDER)) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)

        for i in range(len(FIG5_ORDER)):
            for j in range(len(FIG5_METRICS)):
                ax.text(j, i, f"{mat[i, j]:.4f}", ha="center", va="center",
                        fontsize=8.5, color="black")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Performance Score", fontsize=10, rotation=90)
        cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.ax.tick_params(labelsize=9)

        ax.set_xlabel("Performance Metrics", fontsize=11)
        ax.set_ylabel("Machine Learning Models", fontsize=11)
        ax.set_title(f"{title} (n={n})", fontsize=12, fontweight="bold", pad=12)

        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]:
            ax.spines[sp].set_linewidth(0.8)
            ax.spines[sp].set_color("#888888")

        p = sub / fname
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig] 已保存 {p.name}")

    y_train = np.asarray(C["y_train"])
    p_train = {m: C["proba"][m]["train"] for m in FIG5_ORDER}
    mat_train = _perf_matrix(y_train, p_train)
    _draw(mat_train,
          "A. Prediction performance of different models in the training set",
          "A. Prediction performance of training set.png",
          int(C["n_train"]))

    y_val = np.asarray(C["y_val"])
    p_val = {m: C["proba"][m]["val"] for m in FIG5_ORDER}
    mat_val = _perf_matrix(y_val, p_val)
    _draw(mat_val,
          "B. Prediction performance of different models in the internal validation set",
          "B. Prediction performance of internal validation set.png",
          int(C["n_val"]))

    y_ext = np.asarray(C["y_external"])
    p_ext = {m: C["proba"][m]["ext"] for m in FIG5_ORDER}
    mat_ext = _perf_matrix(y_ext, p_ext)
    _draw(mat_ext,
          "C. Prediction performance of different models in the external validation set",
          "C. Prediction performance of external validation set.png",
          int(C["n_external"]))

def fig6():
    """
    按 Word 文档 Figure 6 参考格式，把 2x2 合并图拆分为独立单图，
    存入 delivery/Figure 6. External CatBoost ROC PR DCA calibration/ 子文件夹：
      A-D: 最终模型 CatBoost 在【外部验证集（完整 485 例）】上的预测
           （ROC / PR / DCA / Calibration；外部队列未参与训练与阈值选定）
    每张图包含 CatBoost 曲线与对应参考线（随机分类器/无技能基线/Treat all/Treat none/Perfect calibration）。
    """
    sub = DELIV / "Figure 6. External CatBoost ROC PR DCA calibration"
    old_png = DELIV / "Figure 6. External CatBoost ROC PR DCA calibration.png"
    if old_png.exists():
        old_png.unlink()
        print(f"[fig6] 已移除旧合并 PNG: {old_png.name}")
    sub.mkdir(parents=True, exist_ok=True)

    xgb_color = "#C77CBA"
    ref_color = "#888888"

    groups = [
        {
            "y": np.asarray(C["y_external"]),
            "p": np.asarray(C["proba"]["CatBoost"]["ext"]),
            "letters": ["A", "B", "C", "D"],
        },
    ]

    def _draw_one_panel(kind, y, p, letter):
        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        prev = y.mean()
        if kind == "roc":
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, lw=2, color=xgb_color,
                    label=f"CatBoost (AUC={auc(fpr, tpr):.4f})")
            ax.plot([0, 1], [0, 1], "--", color=ref_color, lw=1,
                    label="Random classifier")
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("False Positive Rate", fontsize=11)
            ax.set_ylabel("True Positive Rate", fontsize=11)
            title = f"{letter}. ROC Curve - External Validation Set"
            fname = f"{letter}. ROC Curve - External Validation Set.png"
        elif kind == "pr":
            prec, rec, _ = precision_recall_curve(y, p)
            ax.plot(rec, prec, lw=2, color=xgb_color,
                    label=f"CatBoost (AP={auc(rec, prec):.4f})")
            ax.axhline(prev, xmin=0, xmax=1, color=ref_color, ls="--", lw=1,
                       label=f"No skill (Precision={prev:.4f})")
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("Recall", fontsize=11)
            ax.set_ylabel("Precision", fontsize=11)
            title = f"{letter}. Precision-Recall Curve - External Validation Set"
            fname = f"{letter}. Precision-Recall Curve - External Validation Set.png"
        elif kind == "dca":
            pts = np.linspace(0.01, 0.99, 50)
            nb = dca_net_benefit(y, p, pts)
            ax.plot(pts, prev - (1 - prev) * pts / (1 - pts), "--",
                    color=ref_color, lw=1, label="Treat all")
            ax.plot(pts, np.zeros_like(pts), ":", color=ref_color, lw=1,
                    label="Treat none")
            ax.plot(pts, nb, lw=2, color=xgb_color, label="CatBoost")
            ax.set_xlim(-0.02, 1.02)
            ax.set_xlabel("Threshold Probability", fontsize=11)
            ax.set_ylabel("Net Benefit", fontsize=11)
            title = f"{letter}. Decision Curve Analysis - External Validation Set"
            fname = f"{letter}. Decision Curve Analysis - External Validation Set.png"
        else:
            frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
            brier = brier_score_loss(y, p)
            ax.plot([0, 1], [0, 1], "--", color=ref_color, lw=1,
                    label="Perfect calibration")
            ax.plot(mean_pred, frac_pos, "s-", color=xgb_color, lw=2, ms=5,
                    label=f"CatBoost (Brier={brier:.4f})")
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("Mean Predicted Probability", fontsize=11)
            ax.set_ylabel("True Probability", fontsize=11)
            title = f"{letter}. Calibration Curve - External Validation Set"
            fname = f"{letter}. Calibration Curve - External Validation Set.png"

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.grid(True, ls="--", lw=0.5, alpha=0.35)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]:
            ax.spines[sp].set_linewidth(0.8)
            ax.spines[sp].set_color("#888888")
        ax.legend(fontsize=9, loc="best", framealpha=0.95,
                  edgecolor="#AAAAAA", facecolor="white", fancybox=True)
        out = sub / fname
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig] 已保存 {out.name}")

    for g in groups:
        y, p, letters = g["y"], g["p"], g["letters"]
        _draw_one_panel("roc", y, p, letters[0])
        _draw_one_panel("pr", y, p, letters[1])
        _draw_one_panel("dca", y, p, letters[2])
        _draw_one_panel("calibration", y, p, letters[3])

def _draw_shap(sv, fv, title, fname, labels=None):
    """单张 SHAP 摘要图：左=mean|SHAP| 条形重要性，右=beeswarm 散点（每点一个病例，颜色=特征值）。
    labels 为 one-hot 展开后的特征名（缺省回退原始特征列表）。"""
    sv = np.array(sv); fv = np.array(fv)
    labels = list(labels or feats)
    assert len(labels) == sv.shape[1], "SHAP 特征名与贡献矩阵列数不一致"
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    imp = np.abs(sv).mean(axis=0)
    order_i = np.argsort(imp)[::-1]
    axs[0].barh([labels[i] for i in order_i][::-1], imp[order_i][::-1], color="#4C72B0")
    axs[0].set_title("SHAP feature importance (mean |SHAP|)")
    axs[0].set_xlabel("mean |SHAP value|")
    idx = np.random.RandomState(0).choice(len(sv), size=min(800, len(sv)), replace=False)
    for i in range(len(labels)):
        axs[1].scatter(sv[idx, i], np.random.RandomState(i).normal(i, 0.12, len(idx)),
                       c=fv[idx, i], cmap="coolwarm", s=8, alpha=0.5)
    axs[1].set_yticks(range(len(labels))); axs[1].set_yticklabels(labels, fontsize=8)
    axs[1].set_title("SHAP summary (external cohort)")
    axs[1].set_xlabel("SHAP value")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.96])
    save(fig, fname)

def fig7():
    sub = DELIV / "Figure 7. SHAP summary"
    sub.mkdir(parents=True, exist_ok=True)
    if shap_sum_ext is None:
        print("[fig7] 无 SHAP 外部验证集数据，跳过"); return
    _draw_shap(shap_sum_ext, shap_feat_ext,
               "SHAP summary plot of CatBoost (external validation set)",
               "Figure 7. SHAP summary/Figure 7. SHAP summary plot.png",
               labels=shap_names_ext)

def s1fig():
    if perm_imp_ext is None:
        print("[s1fig] 无 perm_imp_ext 数据，跳过"); return
    s1_colors = {
        "XGB": "#C77CBA", "RF": "#EDC948", "DT": "#59A14F", "KNN": "#76B7B2",
        "MLP": "#4E79A7", "SVM": "#E15759", "LR": "#F28E2B", "BNB": "#A0CBE8",
        "LGBM": "#393B79", "CatBoost": "#637939",
    }
    out_dir = DELIV / "Supplement Figure S1. Feature importance"
    out_dir.mkdir(parents=True, exist_ok=True)
    top_n = 6

    def draw_one(m, imp_raw, dataset_label, suffix):
        imp = np.array(imp_raw)
        order_i = np.argsort(imp)[::-1][:top_n]
        top_imp = imp[order_i]
        top_feats = [feats[i] for i in order_i]
        max_imp = top_imp.max()
        norm = top_imp / max_imp * 100 if max_imp > 0 else top_imp * 0
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        y_pos = np.arange(len(norm))
        ax.barh(y_pos[::-1], norm[::-1], color=s1_colors.get(m, "#55A868"))
        ax.set_yticks(y_pos[::-1])
        ax.set_yticklabels(top_feats[::-1], fontsize=9)
        ax.set_xlabel("Normalized Importance (0-100)", fontsize=10)
        ax.set_ylabel("Features", fontsize=10)
        ax.set_title(f"{m} - Feature Importance (Original Features, {dataset_label})", fontsize=12)
        ax.set_xlim(0, 105)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fname = out_dir / f"S1. {m} {suffix}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig] 已保存 {fname.name}")

    for m in ORDER:
        draw_one(m, perm_imp_ext[m], "external validation set", "external validation")

def write_xlsx(path, df):
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Sheet1")
        ws = w.sheets["Sheet1"]
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for col in ws.columns:
            width = max((len(str(c.value)) for c in col if c.value is not None), default=10)
            ws.column_dimensions[col[0].column_letter].width = min(width + 2, 70)
    print(f"[xlsx] 已保存 {path.name}")

def pformat(p):
    if p != p: return "NA"
    return f"{p:.3f}" if p >= 0.001 else "<0.001"

def table1():
    raw = pd.read_csv("artifacts/cleaned/cleaned_internal.csv")
    if "Unnamed: 0" in raw.columns: raw = raw.drop(columns=["Unnamed: 0"])
    target = "Mets at DX-bone"
    g = raw.copy()
    bm = g[g[target] == 1]; nbm = g[g[target] == 0]
    n_all, n1, n0 = len(g), len(bm), len(nbm)

    def pstar(p):
        if p != p: return "NA"
        s = f"{p:.3f}" if p >= 0.001 else "<0.001"
        return s + ("*" if p < 0.05 else "")

    rows = []

    for col, hdr in [("Age", "Age (mean (SD))"),
                     ("Tumor Size (mm)", "Tumor size (mean (SD)) (mm)")]:
        s = pd.to_numeric(g[col], errors="coerce")
        s1 = pd.to_numeric(bm[col], errors="coerce")
        s0 = pd.to_numeric(nbm[col], errors="coerce")
        _, p = sps.ttest_ind(s1, s0, equal_var=False)
        rows.append([hdr, f"{s.mean():.2f} ({s.std():.2f})",
                     f"{s1.mean():.2f} ({s1.std():.2f})", f"{s0.mean():.2f} ({s0.std():.2f})", pstar(p)])

    CATVARS = [
        ("Gender", {0: "Male", 1: "Female"}),
        ("Race", {0: "White", 1: "Black", 2: "Other"}),
        ("Marital status", {0: "Divorced", 1: "Married", 2: "Separated", 3: "Single",
                             4: "Unmarried/domestic partner", 5: "Widowed"}),
        ("Primary site", {0: "Main bronchus", 1: "Upper lobe", 2: "Middle lobe",
                          3: "Lower lobe", 4: "Overlapping lesion"}),
        ("Laterality", {0: "Left", 1: "Right"}),
        ("T stage", {0: "T1", 1: "T2", 2: "T3", 3: "T4"}),
        ("N stage", {0: "N0", 1: "N1", 2: "N2", 3: "N3"}),
        ("Separate Tumor Nodules Ipsilateral Lung",
         {0: "None", 1: "Same lobe", 2: "Different lobe", 3: "Same & different lobes",
          4: "Unknown if same or different lobe"}),
        ("Brain Mets", {0: "No", 1: "Yes"}),
        ("Mets at DX-liver", {0: "No", 1: "Yes"}),
        ("Mets at DX-lung", {0: "No", 1: "Yes"}),
        ("Mets at DX-Distant LN", {0: "No", 1: "Yes"}),
    ]
    for col, lab in CATVARS:
        present = set(pd.to_numeric(g[col], errors="coerce").dropna().unique().tolist())
        levels = [c for c in lab.keys() if c in present]
        ct = pd.crosstab(g[target], pd.to_numeric(g[col], errors="coerce"))
        chi2, p, _, _ = sps.chi2_contingency(ct)
        rows.append([f"{col}, n (%)", "", "", "", pstar(p)])
        for lv in levels:
            ca = int((pd.to_numeric(g[col], errors="coerce") == lv).sum())
            c1 = int((pd.to_numeric(bm[col], errors="coerce") == lv).sum())
            c0 = int((pd.to_numeric(nbm[col], errors="coerce") == lv).sum())
            rows.append([f"  {lab[lv]}", f"{ca} ({100*ca/n_all:.1f}%)",
                         f"{c1} ({100*c1/n1:.1f}%)", f"{c0} ({100*c0/n0:.1f}%)", ""])

    rows += [
        ["Notes", "", "", "", ""],
        ["Other race: American Indian/Alaska Native, Asian/Pacific Islander.", "", "", "", ""],
        ["Abbreviations: mm, millimeter; Mets, metastasis; At DX, at the time of diagnosis; "
         "LN, lymph nodes; *P<0.05.", "", "", "", ""],
    ]

    df = pd.DataFrame(rows, columns=["Characteristic", f"All (n={n_all})",
                                     f"Bone mets (+) (n={n1})", f"No bone mets (n={n0})", "P Value"])
    from openpyxl.styles import Font, Alignment, PatternFill
    with pd.ExcelWriter(DELIV / "Table 1. Cohort characteristics.xlsx", engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Sheet1")
        ws = w.sheets["Sheet1"]
        hdr_fill = PatternFill("solid", fgColor="1F4E79")
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF"); c.fill = hdr_fill
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for r in ws.iter_rows():
            m = r[0].value
            if isinstance(m, str) and (m.endswith(", n (%)") or m in ("Notes",)
                                       or m.startswith(("Other race", "Abbreviations"))):
                r[0].font = Font(bold=True, color="1F4E79")
        for col in ws.columns:
            width = max((len(str(c.value)) for c in col if c.value is not None), default=10)
            ws.column_dimensions[col[0].column_letter].width = min(width + 2, 80)
        ws.freeze_panes = "A2"
    print(f"[xlsx] 已保存 Table 1. Cohort characteristics.xlsx")

def table2():
    """
    Table 2. Clinical characteristics of the training and validation set.
    口径对齐 Table 2.Training and validation set.docx（变量 | Training | Validation | P-value）：
      - 训练/验证为内部随机拆分（按骨转移结局分层抽样）：Training n=6123, Validation n=2625。
      - 变量集与 Table 1 保持一致（含 Race；骨转移为分组变量在表内单列为行）。
      - 连续变量做 Welch t 检验；分类变量做卡方检验（train×val 列联）。
      - 文字标签来自 Supplement Table S2 编码，计数来自 X_train/X_val 原始切割。
    """
    def _load(p):
        d = pd.read_csv(p)
        if "Unnamed: 0" in d.columns: d = d.drop(columns=["Unnamed: 0"])
        return d
    gtr = _load(SPLITS / "X_train.csv"); gtr[target] = _load(SPLITS / "y_train.csv").iloc[:, 0].values
    gva = _load(SPLITS / "X_val.csv");   gva[target] = _load(SPLITS / "y_val.csv").iloc[:, 0].values
    n_tr, n_val = len(gtr), len(gva)

    def pstar(p):
        if p != p: return "NA"
        s = f"{p:.3f}" if p >= 0.001 else "<0.001"
        return s + ("*" if p < 0.05 else "")

    rows = []

    for col, hdr in [("Age", "Age (mean (SD))"),
                     ("Tumor Size (mm)", "Tumor size (mean (SD)) (mm)")]:
        s_t = pd.to_numeric(gtr[col], errors="coerce")
        s_v = pd.to_numeric(gva[col], errors="coerce")
        _, p = sps.ttest_ind(s_t, s_v, equal_var=False)
        rows.append([hdr, f"{s_t.mean():.2f} ({s_t.std():.2f})",
                     f"{s_v.mean():.2f} ({s_v.std():.2f})", pstar(p)])

    CATVARS = [
        ("Gender", {0: "Male", 1: "Female"}),
        ("Race", {0: "White", 1: "Black", 2: "Other"}),
        ("Marital status", {0: "Divorced", 1: "Married", 2: "Separated", 3: "Single",
                            4: "Unmarried/domestic partner", 5: "Widowed"}),
        ("Primary site", {0: "Main bronchus", 1: "Upper lobe", 2: "Middle lobe",
                          3: "Lower lobe", 4: "Overlapping lesion"}),
        ("Laterality", {0: "Left", 1: "Right"}),
        ("T stage", {0: "T1", 1: "T2", 2: "T3", 3: "T4"}),
        ("N stage", {0: "N0", 1: "N1", 2: "N2", 3: "N3"}),
        ("Separate Tumor Nodules Ipsilateral Lung",
         {0: "None", 1: "Same lobe", 2: "Different lobe", 3: "Same & different lobes",
          4: "Unknown if same or different lobe"}),
        ("Brain Mets", {0: "No", 1: "Yes"}),
        ("Mets at DX-liver", {0: "No", 1: "Yes"}),
        ("Mets at DX-lung", {0: "No", 1: "Yes"}),
        ("Mets at DX-Distant LN", {0: "No", 1: "Yes"}),
        ("Mets at DX-bone", {0: "No", 1: "Yes"}),
    ]
    for col, lab in CATVARS:
        present = set(pd.to_numeric(gtr[col], errors="coerce").dropna().unique().tolist()) | \
                  set(pd.to_numeric(gva[col], errors="coerce").dropna().unique().tolist())
        levels = [c for c in lab.keys() if c in present]
        comb = pd.concat([
            pd.DataFrame({"grp": "train", "x": pd.to_numeric(gtr[col], errors="coerce")}),
            pd.DataFrame({"grp": "val",   "x": pd.to_numeric(gva[col], errors="coerce")}),
        ], ignore_index=True)
        ct = pd.crosstab(comb["grp"], comb["x"])
        chi2, p, _, _ = sps.chi2_contingency(ct)
        rows.append([f"{col}, n (%)", "", "", pstar(p)])
        for lv in levels:
            ct_t = int((pd.to_numeric(gtr[col], errors="coerce") == lv).sum())
            ct_v = int((pd.to_numeric(gva[col], errors="coerce") == lv).sum())
            rows.append([f"  {lab[lv]}", f"{ct_t} ({100*ct_t/n_tr:.1f}%)",
                         f"{ct_v} ({100*ct_v/n_val:.1f}%)", ""])

    rows += [
        ["Notes", "", "", ""],
        ["The training set (n=6123) and validation set (n=2625) are an internal random "
         "split (7:3) of the original internal cohort (n=8748); the split random seed was "
         "pre-specified by the study protocol (see MANIFEST).", "", "", ""],
        ["Other race: American Indian/Alaska Native, Asian/Pacific Islander.", "", "", ""],
        ["Race was excluded from model training (see Supplement Table S2).", "", "", ""],
        ["Abbreviations: mm, millimeter; Mets, metastasis; At DX, at the time of diagnosis; "
         "LN, lymph nodes; BM, brain metastasis; *P<0.05.", "", "", ""],
    ]

    df = pd.DataFrame(rows, columns=["Characteristic", f"Training (n={n_tr})",
                                     f"Validation (n={n_val})", "P-value"])

    from openpyxl.styles import Font, Alignment, PatternFill
    with pd.ExcelWriter(DELIV / "Table 2. Train vs validation characteristics.xlsx", engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Sheet1")
        ws = w.sheets["Sheet1"]
        hdr_fill = PatternFill("solid", fgColor="1F4E79")
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF"); c.fill = hdr_fill
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for r in ws.iter_rows():
            m = r[0].value
            if isinstance(m, str) and (m.endswith(", n (%)") or m == "Notes"
                                       or m.startswith(("Other race", "Race was", "Abbreviations",
                                                        "The training set"))):
                r[0].font = Font(bold=True, color="1F4E79")
        for col in ws.columns:
            width = max((len(str(c.value)) for c in col if c.value is not None), default=10)
            ws.column_dimensions[col[0].column_letter].width = min(width + 2, 80)
        ws.freeze_panes = "A2"
    print(f"[xlsx] 已保存 Table 2. Train vs validation characteristics.xlsx ({len(df)} 行)")

def _fit_logit(X, y):
    from statsmodels.api import Logit
    X = X.astype(float)
    Xc = sm.add_constant(X, has_constant="add")
    try:
        return Logit(y, Xc).fit(disp=0, maxiter=300)
    except Exception:
        try:
            return Logit(y, Xc).fit(disp=0, method="bfgs", maxiter=500)
        except Exception:
            return None

def table3():
    """
    Table 3. Univariate and multivariable logistic regression（骨转移结局）。
    口径对齐 Table 3.Univariate and multivariable logistic regression.docx：
      列：Variables | Category | Univariate OR(95% CI) | P | Multivariable OR(95% CI) | P
      - 单因素：每变量单独 logistic 回归；连续变量按 1 单位，分类变量各水平 vs 参考水平。
      - 多因素：全变量（13 特征 + Race）一次性 logistic 回归，报告校正 OR(aOR)。
      - 编码与 Table 1/2、Supplement Table S2 一致：内部队列全量 (n=8748, X_train+X_val 拼接)，
        不使用 cleaned_internal.csv（其 2/7 为未知哨兵，会误删合法水平）。
      - 骨转移为结局，故不作为预测变量行；额外纳入 Brain Mets 等模型特征（参考文档未列但本模型含）。
    """
    def _load(p):
        d = pd.read_csv(p)
        if "Unnamed: 0" in d.columns: d = d.drop(columns=["Unnamed: 0"])
        return d
    g = pd.concat([_load(SPLITS / "X_train.csv"), _load(SPLITS / "X_val.csv")], ignore_index=True)
    yt = pd.concat([_load(SPLITS / "y_train.csv").iloc[:, 0],
                    _load(SPLITS / "y_val.csv").iloc[:, 0]], ignore_index=True)
    g[target] = yt.values.astype(float)
    N = len(g)

    LAB = {
        "Gender": {0: "Male", 1: "Female"},
        "Race": {0: "White", 1: "Black", 2: "Other"},
        "Marital status": {0: "Divorced", 1: "Married", 2: "Separated", 3: "Single",
                           4: "Unmarried/domestic partner", 5: "Widowed"},
        "Primary site": {0: "Main bronchus", 1: "Upper lobe", 2: "Middle lobe",
                         3: "Lower lobe", 4: "Overlapping lesion"},
        "Laterality": {0: "Left", 1: "Right"},
        "T stage": {0: "T1", 1: "T2", 2: "T3", 3: "T4"},
        "N stage": {0: "N0", 1: "N1", 2: "N2", 3: "N3"},
        "Separate Tumor Nodules Ipsilateral Lung":
            {0: "None", 1: "Same lobe", 2: "Different lobe", 3: "Same & different lobes",
             4: "Unknown if same or different lobe"},
        "Brain Mets": {0: "No", 1: "Yes"},
        "Mets at DX-liver": {0: "No", 1: "Yes"},
        "Mets at DX-lung": {0: "No", 1: "Yes"},
        "Mets at DX-Distant LN": {0: "No", 1: "Yes"},
    }
    CONT = {"Age", "Tumor Size (mm)"}
    ORDER_VARS = ["Age", "Tumor Size (mm)", "Gender", "Race", "Marital status", "Primary site",
                  "Laterality", "T stage", "N stage",
                  "Separate Tumor Nodules Ipsilateral Lung", "Brain Mets",
                  "Mets at DX-liver", "Mets at DX-lung", "Mets at DX-Distant LN"]
    REF = {v: (1 if v in ("Marital status", "Primary site") else 0) for v in LAB}

    def build(vars_subset, data):
        frames, names = [], []
        for v in vars_subset:
            if v in CONT:
                frames.append(data[[v]].astype(float).rename(columns={v: v}))
                names.append((v, None))
            else:
                ref = REF[v]
                for lv in sorted(c for c in pd.to_numeric(data[v], errors="coerce").dropna().unique().tolist() if c != ref):
                    frames.append(((data[v].astype(float) == lv).astype(float)).rename(f"{v}={lv}"))
                    names.append((v, lv))
        if not frames: return None, []
        return pd.concat(frames, axis=1), names

    def results(X, y, names):
        res = _fit_logit(X, y)
        out = {}
        if res is None: return out
        for (v, lv) in names:
            col = v if lv is None else f"{v}={lv}"
            if col not in res.params: continue
            coef, se = res.params[col], res.bse[col]
            p = res.pvalues[col]
            out[(v, lv)] = (np.exp(coef), np.exp(coef - 1.96 * se), np.exp(coef + 1.96 * se), p)
        return out

    uni = {}
    for v in ORDER_VARS:
        sub = g[[v, target]].dropna()
        X, names = build([v], sub)
        if X is None or X.shape[1] == 0: continue
        for (vv, lv), val in results(X, sub[target].values.astype(float), names).items():
            uni[(vv, lv)] = val
    sub = g[ORDER_VARS + [target]].dropna()
    Xm, mnames = build(ORDER_VARS, sub)
    multi = {}
    if Xm is not None and Xm.shape[1] > 0:
        for (vv, lv), val in results(Xm, sub[target].values.astype(float), mnames).items():
            multi[(vv, lv)] = val

    def _orstr(t):
        if not t: return ""
        orr, lo, hi, _ = t
        return f"{orr:.3f}({lo:.3f}-{hi:.3f})"
    def _pstr(t):
        if not t: return ""
        _, _, _, p = t
        if p != p: return "NA"
        s = "<0.001" if p < 0.001 else f"{p:.3f}"
        return s + ("*" if p < 0.05 else "")

    rows = []
    for v in ORDER_VARS:
        if v in CONT:
            u, m = uni.get((v, None)), multi.get((v, None))
            rows.append([v, "\\", _orstr(u), _pstr(u), _orstr(m), _pstr(m)])
        else:
            ref = REF[v]
            rows.append([v, LAB[v][ref], "Ref", "Ref", "Ref", "Ref"])
            for lv in sorted(c for c in pd.to_numeric(g[v], errors="coerce").dropna().unique().tolist() if c != ref):
                u, m = uni.get((v, lv)), multi.get((v, lv))
                rows.append([v, LAB[v][lv], _orstr(u), _pstr(u), _orstr(m), _pstr(m)])

    rows += [
        ["Notes", "", "", "", "", ""],
        [f"Data source: internal cohort (n={N}); outcome = bone metastasis. All candidate variables "
         "plus Race were entered as predictors in both analyses.", "", "", "", "", ""],
        ["Race was included in this regression for completeness, but was excluded from the machine-learning "
         "model (see Supplement Table S2). Variables with P<0.05 in the multivariable analysis were used "
         "for machine-learning modeling.", "", "", "", "", ""],
        ["Reference categories: Male, White, Married, Upper lobe, Left, T1, N0, None, No "
         "(for metastatic sites and brain).", "", "", "", "", ""],
        ["Other race: American Indian/Alaska Native, Asian/Pacific Islander.", "", "", "", "", ""],
        ["Abbreviations: Mets, metastasis; At DX, at the time of diagnosis; LN, lymph nodes; "
         "BM, brain metastasis; *P<0.05.", "", "", "", "", ""],
    ]

    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Table 3"
    ws.append(["Variables", "Category", "Univariate analysis", "Univariate analysis",
               "Multivariable analysis", "Multivariable analysis"])
    ws.append(["", "", "Odds Ratio (95% CI)", "P Value", "Odds Ratio (95% CI)", "P Value"])
    for r in rows: ws.append(r)
    ws.merge_cells("A1:A2"); ws.merge_cells("B1:B2"); ws.merge_cells("C1:D1"); ws.merge_cells("E1:F1")
    hdr_fill = PatternFill("solid", fgColor="1F4E79")
    thin = Side(style="thin", color="D9D9D9"); border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for c in ws[1] + ws[2]:
        c.font = Font(bold=True, color="FFFFFF"); c.fill = hdr_fill
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in ws.iter_rows(min_row=3):
        a = row[0].value
        if isinstance(a, str) and (a in LAB or a in CONT or a == "Notes"):
            row[0].font = Font(bold=True, color="1F4E79")
        for c in row: c.border = border
    for i, w in enumerate([30, 44, 22, 12, 22, 12], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A3"
    wb.save(DELIV / "Table 3. Univariate multivariate logistic.xlsx")
    print(f"[xlsx] 已保存 Table 3. Univariate multivariate logistic.xlsx ({len(rows)} 行)")

def _perf_df(which):
    cols = ["Model","AUC","Accuracy","Sensitivity","Specificity","PPV","NPV","F1","MCC","Brier","AP"]
    rows = []
    for m in ORDER:
        d = metrics[m][which]
        rows.append([m, round(d["auc"],3), round(d["accuracy"],3), round(d["sensitivity"],3),
                     round(d["specificity"],3), round(d["ppv"],3), round(d["npv"],3),
                     round(d["f1"],3), round(d["mcc"],3), round(d["brier"],4), round(d["ap"],4)])
    return pd.DataFrame(rows, columns=cols)

def table4():
    """Table 4. 十个 ML 模型（8 经典 + LGBM/CatBoost 强化 GBDT）在内部验证集
    (X_val, n=2625)上的性能对比。
    对齐 Table 4.Performance comparison of eight ML models.docx 的列结构：
      列：Model | AUC | Accuracy | Precision | Recall | F1-Score | MCC | Brier Score
      Precision 取 PPV、Recall 取 Sensitivity；分类指标所用的决策阈值在内部验证集上
      以最大化 F1 选定（best-F1 threshold）后冻结，训练/内部验证/外部验证统一套用。"""
    cols = ["Model", "AUC", "Accuracy", "Precision", "Recall", "F1-Score", "MCC", "Brier Score"]
    rows = []
    for m in ORDER:
        d = metrics[m]["val"]
        rows.append([m, round(d["auc"], 3), round(d["accuracy"], 3), round(d["ppv"], 3),
                     round(d["sensitivity"], 3), round(d["f1"], 3), round(d["mcc"], 3),
                     round(d["brier"], 4)])
    thr_txt = ", ".join(f"{m} = {C['best_th'][m]:.3f}" for m in ORDER)
    rows += [
        ["Notes", "", "", "", "", "", "", ""],
        ["Data source: internal validation cohort (n=2625; X_val). Outcome = bone metastasis. "
         "AUC and Brier score are threshold-independent; the classification metrics (Accuracy, "
         "Precision, Recall, F1-Score, MCC) use the decision threshold selected to maximize F1 on "
         "this internal validation set, then frozen and applied unchanged to all cohorts "
         "(training, internal validation and external validation).",
         "", "", "", "", "", "", ""],
        ["Decision thresholds (per model, internal-validation F1-optimal): " + thr_txt,
         "", "", "", "", "", "", ""],
        ["Abbreviations: ML, machine learning; AUC, area under the curve; MCC, Matthews correlation "
         "coefficient; XGB, extreme gradient boosting; RF, random forest; DT, decision tree; KNN, "
         "K-nearest neighbor; MLP, multilayer perceptron; SVM, support vector machine; LR, logistic "
         "regression; BNB, Bernoulli naive Bayes; LGBM, light gradient boosting machine "
         "(LightGBM); CatBoost, gradient boosting with categorical feature support.",
         "", "", "", "", "", "", "", ""],
    ]
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Table 4"
    ws.append(cols)
    for r in rows: ws.append(r)
    hdr_fill = PatternFill("solid", fgColor="1F4E79")
    thin = Side(style="thin", color="D9D9D9"); border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF"); c.fill = hdr_fill
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in ws.iter_rows(min_row=2):
        a = row[0].value
        is_note = a in ("Notes",) or (isinstance(a, str) and (
            a.startswith("Data source") or a.startswith("Decision thresholds")
            or a.startswith("Abbreviations")))
        if is_note:
            row[0].font = Font(bold=True, color="1F4E79")
        for c in row:
            c.border = border
            if is_note:
                c.alignment = Alignment(wrap_text=True, vertical="top")
    for i, w in enumerate([10, 9, 11, 11, 10, 11, 10, 12], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"
    wb.save(DELIV / "Table 4. ML models performance.xlsx")
    print(f"[xlsx] 已保存 Table 4. ML models performance.xlsx ({len(rows)} 行)")

def table4_external():
    """Table 4 外部验证版（内部训练模型，分别在外部485 / 外部291随机子集上评估）。
    对齐 Table 4.Performance comparison of eight ML models.docx 的 7 列格式（10 模型）；标题区分 485 / 291。
    两个文件输出到 delivery/Table 4. ML models performance/。"""
    from sklearn.metrics import (roc_auc_score, brier_score_loss, accuracy_score,
                                 precision_score, recall_score, f1_score, matthews_corrcoef)
    OUT = DELIV / "Table 4. ML models performance"
    OUT.mkdir(parents=True, exist_ok=True)
    _ex_all = np.arange(len(C["y_external"]))
    tr_idx = np.sort(np.random.default_rng(42).choice(_ex_all, size=291, replace=False))
    y485 = np.array(C["y_external"]); y291 = y485[tr_idx]
    cols = ["Model", "AUC", "Accuracy", "Precision", "Recall", "F1-Score", "MCC", "Brier Score"]

    def build_rows(which):
        rows = []
        for m in ORDER:
            if which == "485":
                d = metrics[m]["ext"]
                rows.append([m, round(d["auc"], 3), round(d["accuracy"], 3), round(d["ppv"], 3),
                             round(d["sensitivity"], 3), round(d["f1"], 3), round(d["mcc"], 3),
                             round(d["brier"], 4)])
            else:
                p = np.array(C["proba"][m]["ext"])[tr_idx]
                y = y291
                auc = roc_auc_score(y, p); brier = brier_score_loss(y, p)
                hard = (p >= C["best_th"][m]).astype(int)
                acc = accuracy_score(y, hard); prec = precision_score(y, hard, zero_division=0)
                rec = recall_score(y, hard, zero_division=0); f1 = f1_score(y, hard, zero_division=0)
                mcc = matthews_corrcoef(y, hard) if len(set(hard)) > 1 else 0.0
                rows.append([m, round(auc, 3), round(acc, 3), round(prec, 3), round(rec, 3),
                             round(f1, 3), round(mcc, 3), round(brier, 4)])
        return rows

    def save(rows, n, suffix):
        wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Table 4"
        title = (f"Table 4. Performance comparison of ML models - evaluated on the "
                 f"external validation cohort (n={n})")
        ws.append([title] + [""] * 7); ws.merge_cells("A1:H1")
        ws.append(cols)
        for r in rows: ws.append(r)
        thr_txt = ", ".join(f"{m} = {C['best_th'][m]:.3f}" for m in ORDER)
        subset_note = (" The n=291 file is a fixed random subsample (numpy default_rng(42)) of the "
                       "full external cohort, provided for sensitivity analysis."
                       if suffix == "291" else "")
        notes = [
            ["Notes", "", "", "", "", "", "", ""],
            [f"Data source: external validation cohort (n={n}). Models were trained on the internal "
             f"cohort (n=6123) and evaluated here without further training (external validation). "
             f"Outcome = bone metastasis. AUC and Brier score are threshold-independent; classification "
             f"metrics use the decision threshold selected to maximize F1 on the internal validation "
             f"set, then frozen and applied unchanged here.{subset_note}",
             "", "", "", "", "", "", ""],
            ["Decision thresholds (per model, internal-validation F1-optimal): " + thr_txt,
             "", "", "", "", "", "", ""],
            ["Abbreviations: ML, machine learning; AUC, area under the curve; MCC, Matthews correlation "
             "coefficient; XGB, extreme gradient boosting; RF, random forest; DT, decision tree; KNN, "
             "K-nearest neighbor; MLP, multilayer perceptron; SVM, support vector machine; LR, logistic "
             "regression; BNB, Bernoulli naive Bayes; LGBM, light gradient boosting machine "
             "(LightGBM); CatBoost, gradient boosting with categorical feature support.",
             "", "", "", "", "", "", "", ""],
        ]
        for r in notes: ws.append(r)
        hdr_fill = PatternFill("solid", fgColor="1F4E79")
        thin = Side(style="thin", color="D9D9D9"); border = Border(left=thin, right=thin, top=thin, bottom=thin)
        ws["A1"].font = Font(bold=True, size=12, color="1F4E79")
        ws["A1"].alignment = Alignment(vertical="center")
        for c in ws[2]:
            c.font = Font(bold=True, color="FFFFFF"); c.fill = hdr_fill
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for row in ws.iter_rows(min_row=3):
            a = row[0].value
            is_note = a in ("Notes",) or (isinstance(a, str) and (
                a.startswith("Data source") or a.startswith("Decision thresholds")
                or a.startswith("Abbreviations")))
            if is_note: row[0].font = Font(bold=True, color="1F4E79")
            for c in row:
                c.border = border
                if is_note: c.alignment = Alignment(wrap_text=True, vertical="top")
        for i, w in enumerate([10, 9, 11, 11, 10, 11, 10, 12], 1):
            ws.column_dimensions[get_column_letter(i)].width = w
        ws.freeze_panes = "A3"
        path = OUT / f"Table 4. ML models performance - external {suffix}.xlsx"
        wb.save(path)
        print(f"[xlsx] 已保存 {path.name} ({len(rows)} 行)")

    save(build_rows("485"), 485, "485")
    save(build_rows("291"), 291, "291")

def _table5_render(g, out_path, cohort_name, n_all, n1, n0, race_note):
    """
    公共渲染逻辑：外部队列（485 或 291）的人群基线特征表。
    g: 已合并结局列 [target] 的外部特征 DataFrame（含 Race）。
    """
    import warnings
    from scipy import stats as sps
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    from openpyxl.utils import get_column_letter
    LAB = {
        "Gender": {0: "Male", 1: "Female"},
        "Race": {0: "White", 1: "Black", 2: "Other"},
        "Marital status": {0: "Divorced", 1: "Married", 2: "Separated", 3: "Single",
                           4: "Unmarried/Domestic", 5: "Widowed", 7: "Unknown"},
        "Primary site": {0: "Main bronchus", 1: "Upper lobe", 2: "Middle lobe",
                         3: "Lower lobe", 4: "Overlapping lesion"},
        "Laterality": {0: "Left", 1: "Right"},
        "T stage": {0: "T1", 1: "T2", 2: "T3", 3: "T4"},
        "N stage": {0: "N0", 1: "N1", 2: "N2", 3: "N3"},
        "Separate Tumor Nodules Ipsilateral Lung":
            {0: "None", 1: "Same lobe", 2: "Different lobe", 3: "Same & different lobes",
             4: "Unknown if same or different lobe", 5: "Not documented"},
        "Brain Mets": {0: "No", 1: "Yes"},
        "Mets at DX-liver": {0: "No", 1: "Yes", 2: "Unknown"},
        "Mets at DX-lung": {0: "No", 1: "Yes", 2: "Unknown"},
        "Mets at DX-Distant LN": {0: "No", 1: "Yes", 2: "Unknown"},
    }
    ORDER = ["Age", "Gender", "Race", "Marital status", "Primary site", "Laterality",
             "T stage", "N stage", "Tumor Size (mm)",
             "Separate Tumor Nodules Ipsilateral Lung", "Brain Mets",
             "Mets at DX-liver", "Mets at DX-lung", "Mets at DX-Distant LN"]

    def pstar(p):
        if p != p: return "NA"
        s = f"{p:.3f}" if p >= 0.001 else "<0.001"
        return s + ("*" if p < 0.05 else "")

    def cont_header(f):
        return "Age (mean (SD))" if f == "Age" else "Tumor size (mean (SD)) (mm)"

    bm = g[g[target] == 1]; nbm = g[g[target] == 0]
    rows, parent_labels = [], set()
    for f in ORDER:
        if f in CONTINUOUS:
            s = pd.to_numeric(g[f], errors="coerce")
            s1 = pd.to_numeric(bm[f], errors="coerce"); s0 = pd.to_numeric(nbm[f], errors="coerce")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, p = sps.ttest_ind(s1, s0, equal_var=False)
            h = cont_header(f); parent_labels.add(h)
            rows.append([h, f"{s.mean():.1f} ± {s.std():.1f}",
                         f"{s1.mean():.1f} ± {s1.std():.1f}", f"{s0.mean():.1f} ± {s0.std():.1f}", pstar(p)])
        else:
            levels = sorted([v for v in g[f].dropna().unique().tolist()])
            ct = pd.crosstab(g[target], g[f])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, p, _, _ = sps.chi2_contingency(ct)
            h = f + ", n (%)"; parent_labels.add(h)
            rows.append([h, "", "", "", pstar(p)])
            for lv in levels:
                lab = LAB.get(f, {}).get(lv, str(lv))
                ca = int((g[f] == lv).sum()); c1 = int((bm[f] == lv).sum()); c0 = int((nbm[f] == lv).sum())
                rows.append(["  " + lab, f"{ca} ({100*ca/n_all:.1f}%)",
                             f"{c1} ({100*c1/n1:.1f}%)", f"{c0} ({100*c0/n0:.1f}%)", ""])

    rows += [
        ["Notes", "", "", "", ""],
        [f"Data source: {cohort_name}; outcome = bone metastasis. "
         f"Bone mets (+) (n={n1}) vs No bone mets (n={n0}).", "", "", "", ""],
        ["Race was included for completeness but excluded from the machine-learning model "
         "(see Supplement Table S2). " + race_note, "", "", "", ""],
        ["Other race: American Indian/Alaska Native, Asian/Pacific Islander.", "", "", "", ""],
        ["Mets at DX-bone is the model outcome and therefore is not listed as a characteristic variable.",
         "", "", "", ""],
        ["Abbreviations: Mets, metastasis; At DX, at the time of diagnosis; LN, lymph nodes; "
         "BM, brain metastasis; mm, millimeter; *P<0.05.", "", "", "", ""],
    ]

    wb = Workbook(); ws = wb.active; ws.title = "Table 5"
    ws.append(["Variables", f"All (n={n_all})", f"Bone mets (+) (n={n1})",
               f"No bone mets (n={n0})", "P Value"])
    for r in rows: ws.append(r)
    hdr_fill = PatternFill("solid", fgColor="1F4E79")
    thin = Side(style="thin", color="D9D9D9"); border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF"); c.fill = hdr_fill
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in ws.iter_rows(min_row=2):
        if row[0].value in parent_labels:
            row[0].font = Font(bold=True, color="1F4E79")
        for c in row: c.border = border
    for i, w in enumerate([44, 16, 20, 20, 12], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"
    wb.save(out_path)
    print(f"[xlsx] 已保存 {out_path} ({len(rows)} 行)")

def table5():
    """
    Table 5. Clinical characteristics of the external validation set (n=485).
    口径对齐 Table 5.Clinical characteristics of external validation set.docx：
      Variables | All (n=485) | Bone mets (+) (n=88) | No bone mets (n=397) | P Value
    """
    Xex_full = pd.read_csv(SPLITS / "X_external.csv")
    if "Unnamed: 0" in Xex_full.columns: Xex_full = Xex_full.drop(columns=["Unnamed: 0"])
    yex_full = pd.read_csv(SPLITS / "y_external.csv")
    if "Unnamed: 0" in yex_full.columns: yex_full = yex_full.drop(columns=["Unnamed: 0"])
    yex_full = yex_full.iloc[:, 0]
    g = Xex_full.copy(); g[target] = yex_full.values.astype(float)
    bm = g[g[target] == 1]; nbm = g[g[target] == 0]
    n_all, n1, n0 = len(g), len(bm), len(nbm)
    race_counts = g["Race"].value_counts().to_dict()
    race_note = "External cohort Race: White %d, Black %d, Other %d." % (
        int(race_counts.get(0, 0)), int(race_counts.get(1, 0)), int(race_counts.get(2, 0)))
    _table5_render(g, DELIV / "Table 5. External cohort characteristics.xlsx",
                   "external validation cohort (n=485)", n_all, n1, n0, race_note)

def table5_external():
    """
    Table 5（外部子集 n=291 版）：外部 485 的固定随机子集
    （default_rng(42) 抽取，与 Table 4 的 291 版本同一口径）。口径与 table5() 完全一致，
    仅队列替换为外部 291；文件名/标题区分 485 与 291。
    """
    Xex_full = pd.read_csv(SPLITS / "X_external.csv")
    if "Unnamed: 0" in Xex_full.columns: Xex_full = Xex_full.drop(columns=["Unnamed: 0"])
    yex_full = pd.read_csv(SPLITS / "y_external.csv")
    if "Unnamed: 0" in yex_full.columns: yex_full = yex_full.drop(columns=["Unnamed: 0"])
    yex_full = yex_full.iloc[:, 0]
    _ex_all = len(Xex_full)
    tr_idx = np.sort(np.random.default_rng(42).choice(_ex_all, size=291, replace=False))
    g = Xex_full.iloc[tr_idx].copy(); g[target] = yex_full.iloc[tr_idx].values.astype(float)
    bm = g[g[target] == 1]; nbm = g[g[target] == 0]
    n_all, n1, n0 = len(g), len(bm), len(nbm)
    race_counts = g["Race"].value_counts().to_dict()
    race_note = "External cohort Race: White %d, Black %d, Other %d." % (
        int(race_counts.get(0, 0)), int(race_counts.get(1, 0)), int(race_counts.get(2, 0)))
    folder = DELIV / "Table 5. External cohort characteristics"
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / "Table 5. External cohort characteristics - external 291.xlsx"
    _table5_render(g, out, "external validation subset (n=291; random subsample of the 485 cohort)",
                   n_all, n1, n0, race_note)

def s1_table():
    FULL_LABELS = {
        "Gender": {0: "Male", 1: "Female"},
        "Race": {0: "White", 1: "Black", 2: "Other"},
        "Marital status": {0: "Divorced", 1: "Married", 2: "Separated", 3: "Single",
                           4: "Unmarried/domestic partner", 5: "Widowed"},
        "Primary site": {0: "Main bronchus", 1: "Upper lobe", 2: "Middle lobe",
                         3: "Lower lobe", 4: "Overlapping lesion"},
        "Laterality": {0: "Left", 1: "Right"},
        "T stage": {0: "T1", 1: "T2", 2: "T3", 3: "T4"},
        "N stage": {0: "N0", 1: "N1", 2: "N2", 3: "N3"},
        "Separate Tumor Nodules Ipsilateral Lung":
            {0: "None", 1: "Same lobe", 2: "Different lobe", 3: "Same & different lobes", 4: "Unknown"},
        "Brain Mets": {0: "No", 1: "Yes"},
        "Mets at DX-liver": {0: "No", 1: "Yes", 2: "Unknown"},
        "Mets at DX-lung": {0: "No", 1: "Yes", 2: "Unknown"},
        "Mets at DX-Distant LN": {0: "No", 1: "Yes", 2: "Unknown"},
    }
    rows_def = [
        ("Basic demographic data", "Age", "Age at diagnosis", "Age", "Continuous",
         "Numeric (years)", "Yes"),
        ("Basic demographic data", "Gender", "Gender", "Gender", "Categorical",
         "Male=0, Female=1", "Yes"),
        ("Basic demographic data", "Race", "Race", "Race", "Categorical",
         "White=0, Black=1, Other=2", "No (excluded from model)"),
        ("Basic demographic data", "Marital status", "Marital status at diagnosis", "Marital status",
         "Categorical", "Divorced=0, Married=1, Separated=2, Single=3, Unmarried or domestic partner=4, Widowed=5",
         "Yes"),
        ("Tumor information", "Primary site", "Primary site of the tumor", "Primary site", "Categorical",
         "Main bronchus=0, Upper lobe=1, Middle lobe=2, Lower lobe=3, Overlapping lesion=4", "Yes"),
        ("Tumor information", "Laterality", "Laterality", "Laterality", "Categorical",
         "Left=0, Right=1", "Yes"),
        ("Tumor information", "TNM stage (T)", "According to AJCC 8th edition", "T stage", "Categorical",
         "T1=0, T2=1, T3=2, T4=3", "Yes"),
        ("Tumor information", "TNM stage (N)", "According to AJCC 8th edition", "N stage", "Categorical",
         "N0=0, N1=1, N2=2, N3=3", "Yes"),
        ("Tumor information", "Tumor size (mm)", "the largest diameter of the primary tumor",
         "Tumor Size (mm)", "Continuous", "Numeric (mm)", "Yes"),
        ("Tumor information", "Show one or more satellite nodules in the same pulmonary lobe",
         "Separate tumor nodules ipsilateral lung", "Separate Tumor Nodules Ipsilateral Lung", "Categorical",
         "None=0, Same lobe=1, Different lobe=2, Same & different lobes=3, Unknown=4", "Yes"),
        ("Tumor information", "Combined with bone metastasis", "Mets at DX-bone", "Mets at DX-bone",
         "Categorical (binary)", "No=0, Yes=1", "Outcome (model target)"),
        ("Tumor information", "Combined with liver metastasis", "Mets at DX-liver", "Mets at DX-liver",
         "Categorical (binary)", "No=0, Yes=1, Unknown=2", "Yes"),
        ("Tumor information", "Combined with lung metastasis", "Mets at DX-lung", "Mets at DX-lung",
         "Categorical (binary)", "No=0, Yes=1, Unknown=2", "Yes"),
        ("Tumor information", "Combined with distant lymph nodes metastasis", "Mets at DX-Distant LN",
         "Mets at DX-Distant LN", "Categorical (binary)", "No=0, Yes=1, Unknown=2", "Yes"),
        ("Tumor information", "Brain metastasis", "Mets at DX-brain", "Brain Mets",
         "Categorical (binary)", "No=0, Yes=1", "Yes"),
    ]

    def cont_stat(col):
        s = pd.to_numeric(Xtr_all[col], errors="coerce")
        return f"{s.mean():.1f} ± {s.std():.1f} (range {s.min():.0f}–{s.max():.0f})"

    def cat_stat(col, labelmap, series):
        vc = series.value_counts()
        parts = []
        for code in sorted(labelmap.keys()):
            if code in vc.index:
                parts.append(f"{labelmap[code]}={int(vc[code])}")
        for code in sorted(vc.index):
            if code not in labelmap:
                parts.append(f"code {int(code)}={int(vc[code])}")
        return "; ".join(parts)

    Xtr_all = load(SPLITS / "X_train.csv")
    stat_rows, norm_rows, role_rows = [], [], []
    for r in rows_def:
        typ, feat = r[4], r[3]
        is_outcome = feat == "Mets at DX-bone"
        in_model = (not is_outcome) and (feat in feats)
        if typ == "Continuous":
            stat = cont_stat(feat)
        elif is_outcome:
            stat = cat_stat(feat, {0: "No", 1: "Yes"}, ytr)
        elif feat in Xtr_all.columns:
            stat = cat_stat(feat, FULL_LABELS.get(feat, {}), Xtr_all[feat])
        else:
            stat = "Not available"
        stat_rows.append(stat)
        if is_outcome:
            role = "Outcome (model target)"
            norm = "Integer code (No=0, Yes=1); outcome variable (not scaled)"
        elif in_model:
            role = "Yes"
            norm = ("Z-score standardization (mean=0, SD=1) fitted on the original training set "
                    "(before oversampling), inside the modeling pipeline"
                    if typ == "Continuous" else
                    "Integer code (Supplement Table S2); then Z-score standardized on the original "
                    "training set (before oversampling), inside the modeling pipeline")
        else:
            role = "No (not selected for modeling)"
            norm = "Not used in the model (descriptive only)"
        norm_rows.append(norm)
        role_rows.append(role)

    cols = ["Category", "Clinical variable", "SEER normalization standard", "Model feature",
            "Type", "Coding scheme (Supplement Table S2)", "Normalization & scaling",
            "Training set (n=6123)", "Included in model"]
    data = [[r[0], r[1], r[2], r[3], r[4], r[5], norm_rows[i], stat_rows[i], role_rows[i]]
            for i, r in enumerate(rows_def)]
    df = pd.DataFrame(data, columns=cols)

    out = DELIV / "Supplement Table S1. Normalization.xlsx"
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Supplement Table S1"
    ws.append(list(df.columns))
    for row in df.itertuples(index=False):
        ws.append(list(row))
    head_fill = PatternFill("solid", fgColor="1F4E78")
    head_font = Font(bold=True, color="FFFFFF", size=10)
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for c in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=c)
        cell.fill = head_fill; cell.font = head_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    widths = [16, 34, 30, 26, 14, 40, 34, 34, 18]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, max_col=ws.max_column):
        for cell in row:
            cell.alignment = Alignment(vertical="center", wrap_text=True)
            cell.border = border
    ws.freeze_panes = "A2"
    wb.save(out)
    print(f"[xlsx] 已保存 {out}  ({ws.max_row-1} 行 × {ws.max_column} 列)")

def s2_table():
    ci = pd.read_csv(ARTS / "cleaned" / "cleaned_internal.csv")
    if "Unnamed: 0" in ci.columns: ci = ci.drop(columns=["Unnamed: 0"])
    N = len(ci)

    DEF = [
        ("Basic demographic data", "Gender", "Gender", [("Male", 0), ("Female", 1)], "Included"),
        ("Basic demographic data", "Age", "Age", "continuous", "Included"),
        ("Basic demographic data", "Race", "Race",
         [("White", 0), ("Black", 1), ("Other (American Indian/AK Native, Asian/Pacific Islander)", 2)],
         "Excluded (not used)"),
        ("Basic demographic data", "Marital status", "Marital status", [
            ("Divorced", 0), ("Married (including common law)", 1), ("Separated", 2),
            ("Single (never married)", 3), ("Unmarried or domestic partner", 4), ("Widowed", 5)], "Included"),
        ("Tumor information", "Primary site", "Primary site", [
            ("Main bronchus", 0), ("Upper lobe", 1), ("Middle lobe", 2),
            ("Lower lobe", 3), ("Overlapping lesion", 4)], "Included"),
        ("Tumor information", "Laterality", "Laterality", [("Left", 0), ("Right", 1)], "Included"),
        ("Tumor information", "T stage", "T stage", [("T1", 0), ("T2", 1), ("T3", 2), ("T4", 3)], "Included"),
        ("Tumor information", "N stage", "N stage", [("N0", 0), ("N1", 1), ("N2", 2), ("N3", 3)], "Included"),
        ("Tumor information", "Separate tumor nodules ipsilateral lung",
         "Separate Tumor Nodules Ipsilateral Lung", [
            ("None; No intrapulmonary mets; Foci in situ/minimally invasive adenocarcinoma", 0),
            ("Separate nodules of same hist type in ipsilateral lung, same lobe", 1),
            ("Separate nodules of same hist type in ipsilateral lung, different lobe", 2),
            ("Separate nodules of same hist type in ipsilateral lung, same AND different lobes", 3),
            ("Separate tumor nodules, ipsilateral lung, unknown if same or different lobe", 4)], "Included"),
        ("Tumor information", "Tumor size (mm)", "Tumor Size (mm)", "continuous", "Included"),
        ("Metastasis at diagnosis", "Brain metastasis", "Brain Mets", [("No", 0), ("Yes", 1)], "Included"),
        ("Metastasis at diagnosis", "Liver metastasis", "Mets at DX-liver", [("No", 0), ("Yes", 1)], "Included"),
        ("Metastasis at diagnosis", "Lung metastasis", "Mets at DX-lung", [("No", 0), ("Yes", 1)], "Included"),
        ("Metastasis at diagnosis", "Distant lymph node metastasis", "Mets at DX-Distant LN",
         [("No", 0), ("Yes", 1)], "Included"),
        ("Metastasis at diagnosis", "Bone metastasis", "Mets at DX-bone", [("No", 0), ("Yes", 1)],
         "Outcome (model target)"),
    ]

    rows = []
    for cat, var, feat, levels, role in DEF:
        if feat == target:
            role = "Outcome (model target)"
        elif feat in feats:
            role = "Included"
        else:
            role = "Excluded (not selected for modeling)"
        if levels == "continuous":
            s = pd.to_numeric(ci[feat], errors="coerce")
            ntxt = f"Range {s.min():.0f}–{s.max():.0f}; mean {s.mean():.1f} ± {s.std():.1f}"
            rows.append([cat, var, feat, "— (continuous)", "Z-score standardized", ntxt, role])
        else:
            for label, code in levels:
                n = int((pd.to_numeric(ci[feat], errors="coerce") == code).sum())
                rows.append([cat, var, feat, label, code, n, role])

    cols = ["Category", "Variable", "Model feature", "Property value", "Encoded value",
            f"n (internal cohort, N={N})", "Role in model"]
    df = pd.DataFrame(rows, columns=cols)

    out = DELIV / "Supplement Table S2. Feature attributes.xlsx"
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Supplement Table S2"
    ws.append(list(df.columns))
    for row in df.itertuples(index=False):
        ws.append(list(row))
    head_fill = PatternFill("solid", fgColor="1F4E78")
    head_font = Font(bold=True, color="FFFFFF", size=10)
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for c in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=c)
        cell.fill = head_fill; cell.font = head_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    widths = [20, 30, 36, 48, 14, 32, 22]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, max_col=ws.max_column):
        for cell in row:
            cell.alignment = Alignment(vertical="center", wrap_text=True)
            cell.border = border
    ws.freeze_panes = "A2"
    wb.save(out)
    print(f"[xlsx] 已保存 {out}  ({ws.max_row-1} 行 × {ws.max_column} 列)")

def s3_table():
    df = _perf_df("train")
    pad = [""] * (len(df.columns) - 1)
    df.loc[len(df)] = ["Notes"] + pad
    df.loc[len(df)] = ["Training-set (n=6123) metrics are in-sample (resubstitution) estimates: "
                       "models were evaluated on the same cases used to fit them, so these numbers "
                       "are optimistic by construction and serve only to document fitting behaviour. "
                       "Out-of-sample performance is reported in Table 4 (internal validation set) "
                       "and the external-validation tables."] + pad
    df.loc[len(df)] = ["Classification metrics use the frozen internal-validation F1-optimal "
                       "thresholds (see Table 4 Notes); AUC/AP/Brier are threshold-independent."] + pad
    write_xlsx(DELIV / "Supplement Table S3. Training performance.xlsx", df)

def s4_table():
    """Supplement Table S4：内部训练集训练的最终模型 CatBoost 在外部验证集上的性能。
    A. 完整外部验证集（n=485）；B. 固定随机子集（n=291）。
    外部队列未参与训练 / 超参搜索 / 阈值选定（阈值来源 = 内部验证集 max-F1）。"""
    from sklearn.metrics import (roc_auc_score, accuracy_score, confusion_matrix,
                                 f1_score, matthews_corrcoef, brier_score_loss,
                                 average_precision_score)
    thr = C["best_th"]["CatBoost"]

    dA = metrics["CatBoost"]["ext"]
    yA = np.array(C["y_external"]); pA = np.array(C["proba"]["CatBoost"]["ext"])

    tr_idx = np.sort(np.random.default_rng(42).choice(np.arange(len(yA)), size=291, replace=False))
    pB = pA[tr_idx]; yB = yA[tr_idx]
    hB = (pB >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yB, hB).ravel()
    dB = {
        "auc": roc_auc_score(yB, pB), "accuracy": accuracy_score(yB, hB),
        "sensitivity": tp/(tp+fn), "specificity": tn/(tn+fp),
        "ppv": tp/(tp+fp), "npv": tn/(tn+fn),
        "f1": f1_score(yB, hB), "mcc": matthews_corrcoef(yB, hB),
        "brier": brier_score_loss(yB, pB), "ap": average_precision_score(yB, pB),
    }

    def block(title, eval_desc, d, pos_rate):
        return [
            [title, ""],
            ["Model", "CatBoost"],
            ["Training set", "Internal training set (n=6123)"],
            ["Evaluation set", eval_desc],
            ["AUC", round(d["auc"],3)], ["Accuracy", round(d["accuracy"],3)],
            ["Sensitivity", round(d["sensitivity"],3)], ["Specificity", round(d["specificity"],3)],
            ["PPV", round(d["ppv"],3)], ["NPV", round(d["npv"],3)], ["F1", round(d["f1"],3)],
            ["MCC", round(d["mcc"],3)], ["Brier", round(d["brier"],4)], ["AP", round(d["ap"],4)],
            ["Threshold (internal-validation F1-optimal)", round(thr,3)],
            ["Positive rate (evaluation set)", round(float(pos_rate),3)],
        ]

    rows = [["Supplement Table S4. External CatBoost performance", ""]]
    rows += block("A. Internal-model XGBoost evaluated on the external validation set (n=485)",
                  "External validation set (n=485)", dA, np.mean(yA))
    rows += [["", ""]]
    rows += block("B. Internal-model XGBoost evaluated on the external 291 random subsample (n=291)",
                  "External validation subset (n=291; default_rng(42) subsample)", dB, np.mean(yB))
    rows += [
        ["Notes", ""],
        ["The external cohort was not involved in model training, hyperparameter search, or "
         "threshold selection. The decision threshold was selected to maximize F1 on the internal "
         "validation set and applied unchanged here.", ""],
        ["Outcome = bone metastasis. External prevalence "
         f"(485) = {np.mean(yA):.3f}; internal training prevalence = {np.mean(C['y_train']):.3f}. "
         "The gap partly reflects population differences between cohorts.", ""],
    ]
    df = pd.DataFrame(rows, columns=["Metric","Value"])

    from openpyxl.styles import Font, Alignment, PatternFill
    with pd.ExcelWriter(DELIV / "Supplement Table S4. External CatBoost performance.xlsx", engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Sheet1")
        ws = w.sheets["Sheet1"]
        hdr_fill = PatternFill("solid", fgColor="1F4E79")
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF"); c.fill = hdr_fill
            c.alignment = Alignment(horizontal="center")
        for r in ws.iter_rows():
            m = r[0].value
            if isinstance(m, str) and (m.startswith(("A. ","B. ","Supplement","Notes"))):
                r[0].font = Font(bold=True, color="1F4E79")
                r[1].font = Font(bold=True, color="1F4E79")
        for col in ws.columns:
            width = max((len(str(c.value)) for c in col if c.value is not None), default=10)
            ws.column_dimensions[col[0].column_letter].width = min(width + 2, 80)
        ws.freeze_panes = "A2"
    print(f"[xlsx] 已保存 Supplement Table S4. External CatBoost performance.xlsx")

def _cal_metrics(y, p):
    """CatBoost 校准表现三指标（在验证集上计算，无泄漏）：
      - 校准截距（calibration-in-the-large，斜率固定为 1，理想值 0）；
      - 校准斜率（logistic 再校准 y ~ logit(p)，理想值 1）；
      - Hosmer–Lemeshow 检验（预测概率十分位分组，df = g − 2）。
    返回 (截距, 斜率, HL χ², 分组数 g, 自由度, P 值)。
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    lp = np.log(p / (1 - p))
    slope_fit = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
    slope = float(slope_fit.params[1])
    inter_fit = sm.Logit(y, np.ones((len(y), 1)), offset=lp).fit(disp=0)
    inter = float(inter_fit.params[0])
    groups = pd.qcut(p, 10, duplicates="drop")
    df_g = pd.DataFrame({"y": y, "p": p, "g": groups})
    hl, ng = 0.0, 0
    for _, d in df_g.groupby("g", observed=True):
        n_i, o_i, pbar = len(d), float(d["y"].sum()), float(d["p"].mean())
        if 0 < pbar < 1:
            hl += (o_i - n_i * pbar) ** 2 / (n_i * pbar * (1 - pbar))
        ng += 1
    dof = max(ng - 2, 1)
    return inter, slope, float(hl), ng, dof, float(sps.chi2.sf(hl, dof))

def s5_table():
    """Supplement Table S5: 最终模型 CatBoost 的校准表现（校准截距/斜率、HL 检验）。
    概率来自 isotonic 校准后的最终 CatBoost 模型；内部验证集与外部验证集均未参与
    训练/调参/校准拟合。"""
    cols = ["Metric", "External validation set (n=485)", "Internal validation set (n=2625)"]
    res = {
        "ext": _cal_metrics(C["y_external"], C["proba"]["CatBoost"]["ext"]),
        "val": _cal_metrics(C["y_val"], C["proba"]["CatBoost"]["val"]),
    }
    def _p(v):
        return "<0.001" if v < 0.001 else f"{v:.3f}"
    rows = [
        ["Calibration intercept (ideal = 0)", f"{res['ext'][0]:.3f}", f"{res['val'][0]:.3f}"],
        ["Calibration slope (ideal = 1)", f"{res['ext'][1]:.3f}", f"{res['val'][1]:.3f}"],
        ["Hosmer–Lemeshow chi-square", f"{res['ext'][2]:.3f}", f"{res['val'][2]:.3f}"],
        ["Hosmer–Lemeshow degrees of freedom", str(res['ext'][4]), str(res['val'][4])],
        ["Hosmer–Lemeshow P value", _p(res['ext'][5]), _p(res['val'][5])],
        ["Notes", "", ""],
        ["Probabilities are from the final CatBoost model after isotonic calibration "
         "(5-fold, fitted within the training set only). Neither validation cohort was involved "
         "in training, hyperparameter search, calibration fitting, or threshold selection.", "", ""],
        ["Calibration intercept was estimated with the slope fixed at 1 (calibration-in-the-large); "
         "calibration slope from logistic recalibration of the outcome on logit(p). "
         "Ideal values: intercept = 0, slope = 1.", "", ""],
        [f"Hosmer–Lemeshow test: deciles of predicted probability (equal-frequency binning; "
         f"g = {res['ext'][3]} groups for external and g = {res['val'][3]} for internal validation "
         f"after collapsing tied bin edges); df = g − 2. P > 0.05 indicates no evidence of "
         f"miscalibration.", "", ""],
    ]
    df = pd.DataFrame(rows, columns=cols)
    write_xlsx(DELIV / "Supplement Table S5. CatBoost calibration.xlsx", df)
    print(f"[xlsx] 已保存 Supplement Table S5. CatBoost calibration.xlsx")
    print(f"[s5] CatBoost 校准：外部 截距={res['ext'][0]:.3f} 斜率={res['ext'][1]:.3f} "
          f"HL={res['ext'][2]:.3f} df={res['ext'][4]} P={_p(res['ext'][5])} | "
          f"内部验证 截距={res['val'][0]:.3f} 斜率={res['val'][1]:.3f} "
          f"HL={res['val'][2]:.3f} df={res['val'][4]} P={_p(res['val'][5])}")

def manifest():
    """生成 delivery/MANIFEST.md：披露每个交付文件的生成脚本、口径与关键参数，
    保证交付物可追溯（切分种子、阈值来源、模型与评估集划分）。"""
    thr_txt = ", ".join(f"{m}={C['best_th'][m]:.3f}" for m in ORDER)
    lines = [
        "# Delivery Manifest",
        "",
        "All deliverables in this folder are generated by `delivery_train.py` (model training & cache)",
        "and `delivery_render.py` (figures & tables) from `artifacts/splits/` produced by the pipeline",
        "(`run.py --steps clean,split`).",
        "",
        "## Key methodology (leakage-free)",
        "",
        f"- Internal cohort: n={N_INT} (training n={N_TR} / internal validation n={N_VAL}); "
        f"external validation cohort: n={N_EX} (never used for training, hyperparameter search, "
        "or threshold selection).",
        "- Split random seed: "
        + (f"fixed `{C.get('split_seed')}` (`splitting.random_state` in config.yaml; the same seed "
           "was used for all reported results)."
           if C.get("split_seed") is not None else "not recorded in cache (fallback `splitting.random_state`)."),
        "- Preprocessing inside an imblearn Pipeline (one-hot encoding of nominal variables "
        "[Marital status, Primary site] -> StandardScaler -> SMOTE -> model); all preprocessing "
        "steps are fitted within each training fold during hyperparameter search and 10-fold CV, so "
        "no synthetic samples or scaling statistics leak across folds.",
        "- Hyperparameters tuned per model by Optuna TPE Bayesian search (40 trials, seed 42); "
        "SMOTE on/off, sampling ratio and k-neighbours were co-tuned with model hyperparameters.",
        "- Probabilities calibrated by isotonic regression (CalibratedClassifierCV, 5-fold on the "
        "training set) after tuning; all reported probabilities/metrics come from the calibrated "
        "models. Calibration is a monotone transform and does not change AUC.",
        f"- Decision thresholds (max-F1 on the internal validation set, frozen): {thr_txt}.",
        "- External-cohort figures/tables (Figure 5C, Figure 6, Figure 7, Supplement Figure S1, "
        "Table 4 external versions, Table 5, Supplement Table S4): internal-trained models evaluated "
        "on the full external cohort (n=485) or its fixed random subsample (n=291, numpy "
        "`default_rng(42)`).",
        "",
        "## Files",
        "",
        "| File | Generated by | Content |",
        "|---|---|---|",
        "| Figure 2. Spearman correlation heatmap.png | fig2() | Spearman matrix, internal cohort |",
        "| Figure 3. AUC of 10-fold cross-validation.png | fig3() | 10-fold CV AUC per model (training set) |",
        "| Figure 4. .../A-H | fig4() | ROC/PR/DCA/calibration, training + internal validation |",
        "| Figure 5. .../A-C | fig5() | metric heatmaps, training / internal validation / external |",
        "| Figure 6. .../A-D | fig6() | internal CatBoost (final model) on external validation set |",
        "| Figure 7. SHAP summary/... | fig7() | SHAP of internal CatBoost (final model) on external validation set |",
        "| Supplement Figure S1. .../ | s1fig() | permutation importance of 8 models, external set |",
        "| Table 1. Cohort characteristics.xlsx | table1() | internal cohort baseline |",
        "| Table 2. Train vs validation characteristics.xlsx | table2() | training vs validation |",
        "| Table 3. Univariate multivariate logistic.xlsx | table3() | uni/multivariate logistic, full internal cohort |",
        "| Table 4. ML models performance.xlsx | table4() | internal validation set (10 models) |",
        "| Table 4. ML models performance/external 485 & 291.xlsx | table4_external() | external validation (full / subsample) |",
        "| Table 5. External cohort characteristics(- external 291).xlsx | table5()/table5_external() | external baseline |",
        "| Supplement Table S1-S4 | s1_table()..s4_table() | normalization / attributes / training perf / external CatBoost |",
        "| Supplement Table S5. CatBoost calibration.xlsx | s5_table() | final-model calibration intercept/slope, HL test |",
        "| Figure 8. Web calculator/ | setup_model.py | CatBoost web demo (deployed uncalibrated model; threshold re-selected on the internal validation set) |",
        "",
    ]
    out = DELIV / "MANIFEST.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[manifest] 已保存 {out}")

if __name__ == "__main__":
    fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); s1fig()
    table1(); table2(); table3(); table4(); table4_external()
    table5(); table5_external()
    s1_table(); s2_table(); s3_table(); s4_table(); s5_table()
    manifest()
    print("[done] 全部交付文件已生成于 delivery/")
