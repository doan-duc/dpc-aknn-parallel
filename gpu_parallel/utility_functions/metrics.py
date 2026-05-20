"""
metrics.py - Tính các chỉ số đánh giá chất lượng phân cụm.

Sử dụng:
    python metrics.py --pred <labels_file.csv> --true <y.csv>

Đầu ra:
    ARI, NMI, ACC in ra màn hình và lưu vào output/benchmarks/
"""
import argparse
import os
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pandas as pd
from datetime import datetime

# ── Đường dẫn ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.join(SCRIPT_DIR, "..", "output", "benchmarks")
os.makedirs(BENCHMARK_DIR, exist_ok=True)


def load_labels(filepath: str) -> np.ndarray:
    """Đọc file CSV 1 cột nhãn (có hoặc không có header 'label')."""
    df = pd.read_csv(filepath, header=None)
    # Bỏ qua dòng header nếu là chữ
    try:
        first = int(df.iloc[0, 0])
    except (ValueError, TypeError):
        df = df.iloc[1:]
    return df.iloc[:, 0].astype(int).values


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Tính ACC bằng thuật toán Hungarian (khớp tối ưu giữa nhãn thật và nhãn dự đoán).
    Đây là phép đo ACC chính xác nhất cho bài toán phân cụm.
    """
    n_classes = max(y_true.max(), y_pred.max()) + 1
    cost_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cost_matrix[t, p] += 1
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    return cost_matrix[row_ind, col_ind].sum() / len(y_true)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Tính đầy đủ ARI, NMI, ACC."""
    return {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic"),
        "ACC": clustering_accuracy(y_true, y_pred),
    }


def print_metrics(metrics: dict, title: str = "Kết quả đánh giá") -> None:
    width = 46
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
    for name, val in metrics.items():
        bar_len = int(val * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {name:5s}: {val:.4f}  [{bar}]")
    print("=" * width + "\n")


def save_metrics(metrics: dict, pred_file: str) -> str:
    """Lưu kết quả vào CSV trong output/benchmarks/."""
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(BENCHMARK_DIR, f"metrics_{ts}.csv")
    rows = [{"metric": k, "value": v, "source": os.path.basename(pred_file)}
            for k, v in metrics.items()]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[metrics.py] Đã lưu kết quả: {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Đánh giá chất lượng phân cụm DPC-AKNN")
    parser.add_argument("--pred", required=True, help="File nhãn dự đoán (output của C)")
    parser.add_argument("--true", required=True, dest="true_labels",
                        help="File nhãn ground-truth")
    parser.add_argument("--save", action="store_true", help="Lưu kết quả vào benchmarks/")
    args = parser.parse_args()

    print(f"[metrics.py] Đọc nhãn dự đoán : {args.pred}")
    print(f"[metrics.py] Đọc nhãn thực    : {args.true_labels}")

    y_pred = load_labels(args.pred)
    y_true = load_labels(args.true_labels)

    if len(y_pred) != len(y_true):
        print(f"[Lỗi] Số lượng nhãn không khớp: pred={len(y_pred)}, true={len(y_true)}")
        sys.exit(1)

    metrics = evaluate(y_true, y_pred)
    print_metrics(metrics, title="DPC-AKNN | CPU Parallel | Fashion-MNIST")

    if args.save:
        save_metrics(metrics, args.pred)


if __name__ == "__main__":
    main()
