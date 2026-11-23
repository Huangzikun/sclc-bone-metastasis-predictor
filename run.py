"""骨转移预测流水线入口

  python run.py --all        # clean -> split -> select -> train
  python run.py --steps clean,split

论文图表另跑 delivery_train.py + delivery_render.py
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from src.config import load_config
from src import cleaning, splitting, feature_selection, training

STEPS = {
    "clean": ("数据清洗", cleaning.run_clean),
    "split": ("数据切割", splitting.run_split),
    "select": ("特征筛选", feature_selection.run_select),
    "train": ("模型训练", training.run_train),
}
ORDER = ["clean", "split", "select", "train"]

def main(argv=None):
    parser = argparse.ArgumentParser(description="骨转移预测工程化流水线")
    parser.add_argument("--all", action="store_true", help="依次运行全部阶段")
    parser.add_argument("--steps", type=str, default=None,
                        help="逗号分隔的阶段名，如 clean,split,select,train")
    parser.add_argument("--config", type=str, default=None,
                        help="自定义配置文件路径（默认 config.yaml）")
    args = parser.parse_args(argv)

    if not args.all and not args.steps:
        parser.print_help()
        return 0

    if args.all:
        requested = list(ORDER)
    else:
        requested = [s.strip() for s in args.steps.split(",") if s.strip()]
        requested = [s for s in ORDER if s in requested]
        if not requested:
            print("未识别到有效阶段，可选：", ", ".join(ORDER))
            return 1

    cfg = load_config(args.config)
    print(f"== 仓库根目录: {cfg.root}")
    print(f"== 将运行阶段: {requested}\n")

    for step in requested:
        label, fn = STEPS[step]
        print(f"\n########## 阶段 [{step}] {label} ##########")
        try:
            fn(cfg)
        except FileNotFoundError as e:
            print(f"[ERROR] 阶段 {step} 缺少前置产物: {e}")
            print("请确认依赖的前序阶段已先运行（clean -> split -> select -> train）。")
            return 1
        except Exception as e:
            print(f"[ERROR] 阶段 {step} 执行失败: {e}")
            raise
    print("\n== 全部请求阶段执行完成。产物位于:", cfg.artifacts_dir)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
