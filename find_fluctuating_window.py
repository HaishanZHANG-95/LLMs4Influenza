"""
从 2017 年起遍历所有可能的 test_end_date，
找出预测窗口（最后 pred_len 行）内目标列有起伏的候选日期。

"有起伏" = 窗口内既有上升也有下降（diff 同时含正值和负值），
           即不是单调递增也不是单调递减。

用法：
    python find_fluctuating_window.py
    python find_fluctuating_window.py --data_path dataset/NorthChina_diff.csv \
                                       --target positive_rate --pred_len 13 \
                                       --start_date 2017-01-01 --top_n 10
"""

import argparse
import pandas as pd
import numpy as np


def find_fluctuating_windows(
    data_path: str = "dataset/NorthChina_diff.csv",
    target: str = "positive_rate",
    pred_len: int = 13,
    start_date: str = "2017-01-01",
) -> list[dict]:
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if target not in df.columns:
        raise ValueError(
            f"列 '{target}' 不存在，可用列：{df.columns.tolist()}"
        )

    values = df[target].values
    dates = df["date"].values
    start_ts = pd.Timestamp(start_date)

    results = []
    for i, d in enumerate(dates):
        if pd.Timestamp(d) < start_ts:
            continue

        end_idx = i + 1  # 包含第 i 行
        if end_idx < pred_len:
            continue

        window = values[end_idx - pred_len : end_idx]
        diffs = np.diff(window)

        has_up   = bool(np.any(diffs > 0))
        has_down = bool(np.any(diffs < 0))

        if has_up and has_down:
            # 计算起伏幅度：极差 + 方向变化次数
            sign_changes = int(np.sum(np.diff(np.sign(diffs)) != 0))
            results.append(
                {
                    "test_end_date": str(pd.Timestamp(d).date()),
                    "range": float(window.max() - window.min()),
                    "sign_changes": sign_changes,       # 方向反转次数，越多越曲折
                    "window_values": [round(float(v), 4) for v in window],
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  default="dataset/NorthChina_diff.csv")
    parser.add_argument("--target",     default="positive_rate")
    parser.add_argument("--pred_len",   type=int, default=13)
    parser.add_argument("--start_date", default="2017-01-01")
    parser.add_argument(
        "--top_n", type=int, default=10,
        help="按起伏幅度排序后打印前 N 个结果（0 = 全部打印）",
    )
    parser.add_argument(
        "--sort_by", default="range",
        choices=["range", "sign_changes"],
        help="排序依据：range=极差最大, sign_changes=方向反转次数最多",
    )
    args = parser.parse_args()

    results = find_fluctuating_windows(
        data_path=args.data_path,
        target=args.target,
        pred_len=args.pred_len,
        start_date=args.start_date,
    )

    print(f"\n从 {args.start_date} 起共找到 {len(results)} 个有起伏的预测窗口\n")
    if not results:
        print("没有符合条件的窗口，请调整 start_date 或 pred_len。")
        return

    # 排序
    results_sorted = sorted(results, key=lambda x: x[args.sort_by], reverse=True)
    show = results_sorted if args.top_n == 0 else results_sorted[: args.top_n]

    print(f"{'test_end_date':<16} {'range':>8} {'sign_changes':>14}  window (last {args.pred_len} values)")
    print("-" * 90)
    for r in show:
        print(
            f"{r['test_end_date']:<16} {r['range']:>8.3f} {r['sign_changes']:>14}  "
            f"{r['window_values']}"
        )

    best = results_sorted[0]
    print(f"\n推荐使用（{args.sort_by} 最大）：")
    print(f"  --test_end_date {best['test_end_date']}")
    print(f"  极差={best['range']:.3f}，方向反转={best['sign_changes']}次")
    print(f"  窗口值：{best['window_values']}")


if __name__ == "__main__":
    main()
