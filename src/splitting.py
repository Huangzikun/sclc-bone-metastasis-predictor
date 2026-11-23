"""数据切割：内部 7:3 切 train/val，外部整体留作外部验证"""
from __future__ import annotations

from sklearn.model_selection import train_test_split

from .config import Config
from .io_utils import log, read_table, save_table

def run_split(cfg: Config) -> dict:
    sp = cfg.get("splitting", default={})
    test_size = sp.get("test_size", 0.3)
    random_state = sp.get("random_state", 42)
    files = sp.get("files", {})

    in_path = cfg.artifacts_dir / "cleaned" / "cleaned_internal.csv"
    out_path = cfg.artifacts_dir / "cleaned" / "cleaned_external.csv"
    in_df = read_table(in_path)
    out_df = read_table(out_path)
    log("split", f"内部清洗数据: {in_df.shape}, 外部清洗数据: {out_df.shape}")

    target = cfg.get("project", "target", default="Brain Mets")

    X_train, X_val = train_test_split(in_df, test_size=test_size, random_state=random_state)
    y_train = X_train[[target]].copy()
    y_val = X_val[[target]].copy()
    X_train = X_train.drop(columns=[target])
    X_val = X_val.drop(columns=[target])

    y_ext = out_df[[target]].copy()
    X_ext = out_df.drop(columns=[target])

    log("split", f"训练集 {X_train.shape}, 内部验证集 {X_val.shape}, 外部验证集 {X_ext.shape}")

    for d in (X_train, y_train, X_val, y_val, X_ext, y_ext):
        d.reset_index(drop=True, inplace=True)

    out_dir = cfg.artifacts_dir / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "X_train": out_dir / files.get("X_train", "X_train.csv"),
        "y_train": out_dir / files.get("y_train", "y_train.csv"),
        "X_val": out_dir / files.get("X_val", "X_val.csv"),
        "y_val": out_dir / files.get("y_val", "y_val.csv"),
        "X_external": out_dir / files.get("X_external", "X_external.csv"),
        "y_external": out_dir / files.get("y_external", "y_external.csv"),
    }
    save_table(X_train, paths["X_train"])
    save_table(y_train, paths["y_train"])
    save_table(X_val, paths["X_val"])
    save_table(y_val, paths["y_val"])
    save_table(X_ext, paths["X_external"])
    save_table(y_ext, paths["y_external"])
    log("split", f"已保存切分结果 -> {out_dir}")
    return {k: str(v) for k, v in paths.items()}
