"""
visualize.py - So sánh Ground-Truth vs Kết quả C binary (DPC-AKNN).

Sử dụng:
    python visualize.py --pred <cpu_labels.csv> --true <y.csv> --data <X.csv>

Tự động chọn:
  - Mặc định sử dụng t-SNE 2D để cho ra hình ảnh đẹp và trực quan nhất.
  - Lưu ý: t-SNE với 70K điểm có thể tốn vài phút để chạy.
"""
import argparse, os, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

COLORS = ["#FF4136","#0074D9","#2ECC40","#FF851B","#B10DC9",
          "#7FDBFF","#FFDC00","#F012BE","#01FF70","#AAAAAA"]
BG = "#0D1117"; FG = "#E6EDF3"
FASHION = ["T-shirt","Trouser","Pullover","Dress","Coat",
           "Sandal","Shirt","Sneaker","Bag","Ankle boot"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "..", "output", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def read_int_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try: rows.append(int(line.strip().split(",")[0]))
            except ValueError: pass
    return np.array(rows, dtype=np.int32)


def read_data_csv(path, max_rows=None):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows: break
            try: rows.append([float(v) for v in line.strip().split(",")])
            except ValueError: pass
    return np.array(rows, dtype=np.float32)


def reduce_2d(X, method="auto"):
    n = len(X)
    use_pca = (method == "pca") or (method == "auto" and n > 10000)
    if use_pca:
        print(f"[viz] PCA 2D (n={n})...")
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(X), "PCA"
    else:
        print(f"[viz] t-SNE 2D (n={n}) truc tiep tren khong gian goc...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
        return tsne.fit_transform(X), "t-SNE"


def compute_metrics(y_true, y_pred, n_cls):
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")
    n   = max(int(y_true.max()), int(y_pred.max())) + 1
    C   = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n and 0 <= p < n: C[t, p] += 1
    r, c = linear_sum_assignment(-C)
    acc  = C[r, c].sum() / len(y_true)
    return {"ARI": ari, "NMI": nmi, "ACC": acc}


def scatter_panel(ax, X2d, labels, n_cls, title, class_names=None):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(colors="#555", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363D")

    for c in range(n_cls):
        mask = labels == c
        if not mask.any(): continue
        ax.scatter(X2d[mask, 0], X2d[mask, 1],
                   c=COLORS[c % 10], s=4, alpha=0.75, linewidths=0)

    ns = class_names or [f"Cum {i}" for i in range(n_cls)]
    patches = [mpatches.Patch(color=COLORS[i % 10],
                              label=ns[i] if i < len(ns) else f"C{i}")
               for i in range(n_cls)]
    ax.legend(handles=patches, loc="lower left", fontsize=6.5,
              framealpha=0.3, labelcolor="white",
              facecolor="#161B22", edgecolor="#30363D", ncol=2)


def main():
    ap = argparse.ArgumentParser(description="Visualize GT vs DPC-AKNN Prediction")
    ap.add_argument("--pred",     required=True, help="File nhan du doan (output C binary)")
    ap.add_argument("--true",     required=True, help="File nhan ground-truth")
    ap.add_argument("--data",     required=True, help="File du lieu X.csv")
    ap.add_argument("--clusters", type=int, default=10)
    ap.add_argument("--method",   default="tsne", choices=["auto","pca","tsne"],
                    help="Phuong phap giam chieu (mac dinh: tsne)")
    ap.add_argument("--maxn",     type=int, default=70000,
                    help="So mau toi da doc (mac dinh 70000)")
    args = ap.parse_args()

    # 1. Đọc dữ liệu
    print(f"[viz] Doc X tu {args.data}...")
    X      = read_data_csv(args.data, max_rows=args.maxn)
    y_true = read_int_csv(args.true)[:len(X)]
    y_pred = read_int_csv(args.pred)[:len(X)]
    n      = len(X)
    print(f"[viz] n={n}, GT={len(y_true)}, Pred={len(y_pred)}")

    # 2. Giảm chiều
    X2d, method_name = reduce_2d(X, args.method)

    # 3. Tính metric
    metrics = compute_metrics(y_true, y_pred, args.clusters)
    print(f"\n{'='*50}")
    print(f"  KET QUA DANH GIA - DPC-AKNN CPU Parallel")
    print(f"  n={n}  |  clusters={args.clusters}")
    print(f"{'='*50}")
    for nm, v in metrics.items():
        bar = "#"*int(v*20) + "-"*(20-int(v*20))
        print(f"  {nm}: {v:.4f}  [{bar}]")
    print(f"{'='*50}\n")

    # 4. Vẽ
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor=BG)
    fig.suptitle(
        f"DPC-AKNN  |  CPU Parallel (OpenMP)  |  Fashion-MNIST  |  {method_name}  |  n={n:,}",
        color=FG, fontsize=13, fontweight="bold"
    )

    scatter_panel(axes[0], X2d, y_true, args.clusters,
                  "Ground-Truth  (Nhan thuc te)", FASHION)

    # Panel phải: Prediction + metrics
    scatter_panel(axes[1], X2d, y_pred, args.clusters,
                  "DPC-AKNN Prediction  (C + OpenMP)")

    # Ghi metric lên góc phải hình
    metric_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    axes[1].text(0.02, 0.98, metric_text,
                 transform=axes[1].transAxes, va="top", ha="left",
                 color="#2ECC40", fontsize=9, fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#161B22",
                           edgecolor="#30363D", alpha=0.85))

    plt.tight_layout()

    ts   = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(PLOTS_DIR, f"comparison_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"[viz] Luu anh: {path}")
    plt.show()


if __name__ == "__main__":
    main()
