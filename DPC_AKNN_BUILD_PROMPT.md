# MASTER PROMPT — Implement DPC-AKNN Algorithm (3 Versions)
# Source: "Density peak clustering based on nearest neighbors"
# Authors: Houshen Lin, Jian Hou, Huaqiang Yuan
# Journal: Engineering Applications of Artificial Intelligence 160 (2025) 111981

---

## OVERVIEW

You are an expert in clustering algorithms, scientific computing, and parallel programming
(OpenMP / CUDA C). Your task is to implement the **DPC-AKNN** algorithm completely and
accurately following the original paper, in three independent language stacks:

| Folder | Language | Parallelism |
|--------|----------|-------------|
| `original/` | Python + NumPy | None (reference implementation) |
| `cpu_parallel/` | C99 + OpenMP | CPU multi-threading via `#pragma omp` |
| `gpu_parallel/` | CUDA C (`.cu`) | GPU via custom CUDA kernels |

```
project/
├── data/                  ← SHARED input data for all 3 folders
│   └── real/              ← Real datasets (.csv, .data) — download and place here
│
├── theory/
│   └── THEORY.md          ← Full theoretical background (see Section A)
│
├── original/              ← python run_demo.py
│   ├── config.py
│   └── outputs/
│
├── cpu_parallel/          ← make && ./dpc_aknn_cpu
│   ├── config.h
│   ├── Makefile
│   └── outputs/
│
└── gpu_parallel/          ← make && ./dpc_aknn_gpu
    ├── config.h
    ├── Makefile
    └── outputs/
```

> **Data is shared, outputs are separate.**
> All three versions read from the same `../data/` directory so benchmark comparisons
> are valid. Each folder writes results, labels, logs, and timing to its own `outputs/`.

---

## MANDATORY CODE RULES

### 1. Comment Language
**ALL comments and docstrings must be written in Vietnamese**, regardless of the
source language (Python `#` / `"""`, C `//` / `/* */`, CUDA `//` / `/* */`).

Every function/kernel must document:
- Mục đích (purpose)
- Bài báo (paper equation / algorithm line)
- Song song hóa: "Không song song" / "OpenMP parallel for" / "CUDA kernel"
- Tham số (parameters with types and shapes)
- Trả về (return value with type)

**Python docstring template:**
```python
def compute_dc(knn_distances: np.ndarray) -> float:
    """
    Tính khoảng cách ngưỡng d_c toàn cục theo Eq. (7), (8), (9).

    Mục đích: Xác định d_c thích nghi theo phân phối kNN cục bộ,
              thay vì percentile cố định như DPC gốc.

    Bài báo: Section 3.1, Eq. (6) → (8) → (9) → (7).
    Song song hóa: Không song song (O(nk), đủ nhanh).

    Tham số:
        knn_distances (np.ndarray): shape (n, k), dtype float64.
    Trả về:
        float: giá trị d_c toàn cục theo Eq. (7).
    """
```

**C / CUDA comment template:**
```c
/*
 * compute_dc — Tính khoảng cách ngưỡng d_c toàn cục.
 *
 * Mục đích: Xác định d_c thích nghi theo phân phối kNN cục bộ.
 * Bài báo:  Section 3.1, Eq. (6) → (8) → (9) → (7).
 * Song song hóa: OpenMP parallel for trên vòng lặp tính d_ci.
 *
 * Tham số:
 *   knn_distances: Ma trận khoảng cách kNN, shape (n, k), double*.
 *   n, k:          Kích thước dữ liệu.
 * Trả về:
 *   double: Giá trị d_c toàn cục.
 */
double compute_dc(const double* knn_distances, int n, int k);
```

### 2. Config Files
- `original/` uses `config.py` (Python constants)
- `cpu_parallel/` and `gpu_parallel/` use `config.h` (C preprocessor `#define`)
- **Algorithm parameters must be identical across all three configs** for fair comparison
- **`DATA_DIR` must point to the shared `../data/` directory** in all three configs

### 3. Fidelity Priority
Implement **100% faithfully to the paper**. Document every simplification in `CHANGELOG.md`.

---

## SECTION A — THEORETICAL BACKGROUND (save to `theory/THEORY.md`)

### A.1. Motivation

The original DPC algorithm (Rodriguez & Laio, 2014) has two problems on real data:
1. **Density kernel sensitivity**: Cutoff kernel ignores distance information; Gaussian
   kernel uses all points including distant ones.
2. **Error propagation during assignment**: One-step assignment propagates label errors.

DPC-AKNN solves both by leveraging kNN at every major step.

---

### A.2. New Density Kernel (Section 3.1)

#### A.2.1. kNN-Constrained Gaussian Kernel — Equation (5)

```
ρ_i = Σ_{x_j ∈ N^k_{x_i}}  exp( −d²_ij / d²_c )          ... (5)
```
- `N^k_{x_i}`: k nearest neighbors of x_i (excluding x_i itself)
- `d_ij`: Euclidean distance between x_i and x_j
- `d_c`: global cutoff distance (see A.2.2)

#### A.2.2. Adaptive d_c — Equations (6), (7), (8), (9)

**Step 1** — Per-point d_ci (Eq. 6):
```
d_ci = d̄_i + σ_i                                           ... (6)
```
- `d̄_i`: mean distance from x_i to its k nearest neighbors
- `σ_i`: standard deviation of those distances
- d_ci > 84% of kNN distances under Gaussian assumption → reduces outlier influence

**Step 2** — Global mean (Eq. 8):
```
d̄_c = (1/n) Σ_{i} d_ci                                     ... (8)
```

**Step 3** — Global std (Eq. 9):
```
σ_c = sqrt( (1/(n−1)) Σ_{i} (d_ci − d̄_c)² )               ... (9)
```

**Step 4** — Global cutoff (Eq. 7):
```
d_c = d̄_c + σ_c                                            ... (7)
```

---

### A.3. Cluster Center Identification

#### A.3.1. Distance to Higher-Density Neighbor — Equation (4)
```
δ_i = min_{j: ρ_j > ρ_i}  d_ij                             ... (4)
```
For the point with the highest density: `δ_i = max_j d_ij` (farthest point in dataset).

#### A.3.2. Decision Index γ
```
γ_i = ρ_i × δ_i
```
Select the `n_c` points with the largest γ as cluster centers.

---

### A.4. Non-Center Data Assignment (Section 3.2)

#### A.4.1. Build Initial Clusters (Section 3.2.1)

**Initialization**: For cluster center c_j of cluster C_j:
- Add c_j and all of N^k_{c_j} to C_j
- Compute centroid φ_j

**Expansion**: For each x_p ∈ N^k_{c_j}, add x_q ∈ N^k_{x_p} to C_j iff all 3 hold:

1. `x_q` not yet assigned
2. `d_pq ≤ (1/k) × Σ_{t ∈ N^k_{x_p}} d_pt` — local distance condition
3. `d(φ_j, x_q) ≤ d_c` — centroid distance condition (uses global d_c, not d_ci)

#### A.4.2. Low-Density Assignment — Association Matrix (Section 3.2.2)

For the remaining `n_u` unassigned points, build `A ∈ R^{n_u × n_c}`:

```
A(i, r) = Σ_{x_l ∈ N^k_{x_i} ∩ C_r}  (1/d_il) × ρ_l × ρ_i   ... (10)
```

Assignment loop:
```
while any nonzero element in A:
    (i*, r*) ← argmax A
    assign x_{i*} to C_{r*}
    set row i* of A to 0
    update A: for each unassigned x_m where x_{i*} ∈ N^k_{x_m}:
        A(m, r*) += (1/d_{m,i*}) × ρ_{i*} × ρ_m              ... (11)
```
Only column r* is updated in Eq. (11).

#### A.4.3. Error Correction — kNN Voting (Section 3.2.3)

Sort all assigned points by density descending. For each x_i:
- Count cluster occurrences in N^k_{x_i}: `h_1, …, h_{n_c}`
- Reassign x_i to cluster with largest `h_r`
- Tie-break: smallest mean distance from x_i to that cluster's points

#### A.4.4. Remaining Points — Equation (12)

```
r* = argmin_r  ζ_ir                                          ... (12)
```
where `ζ_ir` = mean distance from x_i to its k nearest neighbors **within cluster r**.

---

### A.5. Computational Complexity (Section 3.3)

| Step | Line in Alg. 1 | Complexity |
|------|----------------|------------|
| kNN search | Preprocessing | O(n log n) |
| Compute d_c | Line 1 | O(nk) |
| Compute ρ, δ | Line 2 | O(nk) |
| Select centers | Lines 3–4 | O(n) |
| Build initial clusters | Lines 5–15 | O(n_c × k²) |
| Build matrix A | Line 16 | O(n_c × n × k) |
| Assignment loop | Lines 17–22 | **O(n_c × n²) ← bottleneck** |
| Correction & remaining | Lines 23–24 | O(n_c × n) |
| **Total** | | **O(n_c × n²)** |

Recommended k: 10 ≤ k ≤ 30 (from paper experiments on 50 datasets).

---

## SECTION B — CONFIG FILES

### B.1. `original/config.py`

```python
"""
Cấu hình cho phiên bản ORIGINAL (Python / NumPy).
Đây là config chuẩn — các tham số thuật toán phải khớp với config.h
của cpu_parallel/ và gpu_parallel/.

Chạy: cd original/ && python run_demo.py
"""

import os

# ============================================================
# ĐƯỜNG DẪN
# ============================================================

FOLDER_ROOT = os.path.dirname(os.path.abspath(__file__))

# Dữ liệu dùng CHUNG — trỏ về project root/data/
DATA_DIR = os.path.join(FOLDER_ROOT, "..", "data")

# Output RIÊNG của original/
OUTPUT_DIR    = os.path.join(FOLDER_ROOT, "outputs")
PLOTS_DIR     = os.path.join(OUTPUT_DIR, "plots")
LABELS_DIR    = os.path.join(OUTPUT_DIR, "labels")
BENCHMARK_DIR = os.path.join(OUTPUT_DIR, "benchmarks")
LOG_DIR       = os.path.join(OUTPUT_DIR, "logs")

# ============================================================
# THAM SỐ THUẬT TOÁN
# (PHẢI khớp với #define trong cpu_parallel/config.h và gpu_parallel/config.h)
# ============================================================

DEFAULT_N_CLUSTERS = 3
DEFAULT_K          = 15

# ============================================================
# BENCHMARK & ĐÁNH GIÁ
# ============================================================

BENCHMARK_N_REPEATS = 3
EVAL_METRICS       = ["ACC", "ARI", "NMI"]

# ============================================================
# LOGGING & DEMO
# ============================================================

LOG_LEVEL      = "INFO"
LOG_TO_FILE    = True

# ============================================================
# KHỞI TẠO THƯ MỤC OUTPUT (data/ là chung, không tạo ở đây)
# ============================================================
for _dir in [OUTPUT_DIR, PLOTS_DIR, LABELS_DIR, BENCHMARK_DIR, LOG_DIR]:
    os.makedirs(_dir, exist_ok=True)
```

---

### B.2. `cpu_parallel/config.h`

```c
/*
 * config.h — Cấu hình cho phiên bản CPU PARALLEL (C99 + OpenMP).
 *
 *         PHẢI khớp với original/config.py và gpu_parallel/config.h
 *         để đảm bảo so sánh thời gian thực hiện công bằng.
 *
 * DATA_DIR trỏ về thư mục data/ CHUNG ở project root.
 * Cả 3 phiên bản đọc cùng một file dữ liệu — đây là điều kiện để
 * benchmark có giá trị so sánh.
 *
 * Build: cd cpu_parallel/ && make
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <limits.h>

/* ============================================================
 * ĐƯỜNG DẪN
 * Tất cả đường dẫn tương đối với thư mục cpu_parallel/
 * ============================================================ */

/* Dữ liệu dùng CHUNG — trỏ về project root/data/ */
#define DATA_DIR          "../data"

/* Output RIÊNG của cpu_parallel/ */
#define OUTPUT_DIR        "outputs"
#define PLOTS_DIR         "outputs/plots"
#define LABELS_DIR        "outputs/labels"
#define BENCHMARK_DIR     "outputs/benchmarks"
#define LOG_DIR           "outputs/logs"

/* ============================================================
 * THAM SỐ THUẬT TOÁN DPC-AKNN
 * PHẢI khớp với original/config.py và gpu_parallel/config.h
 * ============================================================ */

#define DEFAULT_N_CLUSTERS  3
#define DEFAULT_K           15

/* ============================================================
 * THAM SỐ BENCHMARK
 * PHẢI khớp với original/config.py
 * ============================================================ */

#define BENCHMARK_N_REPEATS  3

/* Các kích thước dataset benchmark — khai báo trong main.c hoặc benchmark.c:
 * int benchmark_sizes[] = {500, 1000, 3000, 5000};
 * int benchmark_n_sizes = 4;
 */

/* ============================================================
 * THAM SỐ OPENMP
 * ============================================================ */

/* Số thread OpenMP
 *  0 = dùng tất cả core (OMP_NUM_THREADS từ môi trường)
 *  N = cố định N thread
 */
#define OMP_N_THREADS       0

/* Kích thước chunk cho schedule(dynamic, CHUNK_SIZE)
 * Dùng cho các vòng lặp không đồng đều (ví dụ: tính δ_i)
 */
#define OMP_CHUNK_SIZE      64

/* ============================================================
 * KIỂU DỮ LIỆU
 * ============================================================ */

/* Dùng double (float64) cho tính toán trên CPU */
typedef double real_t;

/* ============================================================
 * LOGGING
 * ============================================================ */

/* Mức log: 0=ERROR, 1=INFO, 2=DEBUG */
#define LOG_LEVEL  1

#endif /* CONFIG_H */
```

---

### B.3. `gpu_parallel/config.h`

```c
/*
 * config.h — Cấu hình cho phiên bản GPU PARALLEL (CUDA C).
 *
 * LƯU Ý: Tham số thuật toán PHẢI khớp với original/config.py
 *         và cpu_parallel/config.h để so sánh thời gian công bằng.
 *
 * DATA_DIR trỏ về thư mục data/ CHUNG ở project root.
 *
 * Yêu cầu: GPU NVIDIA, CUDA Compute Capability ≥ 3.5, CUDA Toolkit ≥ 11.0
 * Build:   cd gpu_parallel/ && make
 */

#ifndef CONFIG_H
#define CONFIG_H

/* ============================================================
 * ĐƯỜNG DẪN
 * ============================================================ */

/* Dữ liệu dùng CHUNG — trỏ về project root/data/ */
#define DATA_DIR          "../data"

/* Output RIÊNG của gpu_parallel/ */
#define OUTPUT_DIR        "outputs"
#define PLOTS_DIR         "outputs/plots"
#define LABELS_DIR        "outputs/labels"
#define BENCHMARK_DIR     "outputs/benchmarks"
#define LOG_DIR           "outputs/logs"

/* ============================================================
 * THAM SỐ THUẬT TOÁN DPC-AKNN
 * PHẢI khớp với original/config.py và cpu_parallel/config.h
 * ============================================================ */

#define DEFAULT_N_CLUSTERS  3
#define DEFAULT_K           15

/* ============================================================
 * THAM SỐ BENCHMARK
 * ============================================================ */

#define BENCHMARK_N_REPEATS  3

/* GPU có thể xử lý dataset lớn hơn — khai báo thêm trong benchmark.cu:
 * int benchmark_sizes_large[] = {10000, 30000, 50000};
 */

/* ============================================================
 * THAM SỐ CUDA
 * ============================================================ */

/* ID GPU sử dụng (0 = GPU đầu tiên) */
#define GPU_DEVICE_ID       0

/* Kích thước tile cho kernel tính ma trận khoảng cách (tiled shared memory)
 * Phải là bội số của 32 (warp size). Thường: 16 hoặc 32.
 */
#define TILE_SIZE           32

/* Kích thước block 1D cho kernel tính ρ và A
 * Phải là bội số của 32. Thường: 128, 256, hoặc 512.
 */
#define BLOCK_SIZE_1D       256

/* Kích thước block 2D cho kernel tính A(i, r)
 * BLOCK_X × BLOCK_Y = tổng số thread mỗi block (thường ≤ 1024)
 */
#define BLOCK_SIZE_2D_X     16
#define BLOCK_SIZE_2D_Y     16

/* Số điểm xử lý mỗi batch khi tính D để tránh OOM
 * batch_size × n × sizeof(float) = bộ nhớ GPU cần cho 1 batch
 */
#define GPU_BATCH_SIZE      5000

/* Tự động fallback về CPU nếu không tìm thấy GPU (1=có, 0=không) */
#define GPU_FALLBACK_CPU    1

/* ============================================================
 * KIỂU DỮ LIỆU
 * GPU tối ưu với float (float32) — khác với CPU dùng double
 * ============================================================ */

typedef float real_t;

/* ============================================================
 * LOGGING
 * ============================================================ */

#define LOG_LEVEL  1

#endif /* CONFIG_H */
```

---

## SECTION C — DETAILED FOLDER STRUCTURE

```
project/
│
├── data/                        ← SHARED — tất cả 3 folder đọc tại đây
│   └── real/
│
├── theory/
│   └── THEORY.md
│
├── original/                    ← Python / NumPy
│   ├── config.py
│   ├── dpc_aknn.py              ← class DPCAKNN
│   ├── utils.py                 ← 11 hàm tiện ích
│   ├── run_demo.py
│   ├── benchmark.py
│   ├── test_dpc_aknn.py         ← 7 unit tests (pytest)
│   ├── requirements.txt
│   ├── README.md
│   └── outputs/
│       ├── plots/
│       ├── labels/
│       ├── benchmarks/
│       └── logs/
│
├── cpu_parallel/                ← C99 + OpenMP
│   ├── config.h
│   ├── Makefile
│   ├── dpc_aknn.h               ← struct DPCAKNN + function declarations
│   ├── dpc_aknn.c               ← fit(), fit_predict(), save_labels()
│   ├── utils_omp.h
│   ├── utils_omp.c              ← 11 hàm tiện ích, song song hóa OpenMP
│   ├── csv_io.h / csv_io.c      ← đọc/ghi CSV thuần C
│   ├── test_dpc_aknn.c          ← 7 unit tests (tự viết, không dùng framework)
│   ├── README.md
│   └── outputs/
│       ├── labels/
│       ├── benchmarks/
│       └── logs/
│
└── gpu_parallel/                ← CUDA C
    ├── config.h
    ├── Makefile
    ├── main.cu                  ← entry point
    ├── dpc_aknn.h
    ├── dpc_aknn.cu              ← fit(), fit_predict(), save_labels()
    ├── kernels.h
    ├── kernels.cu               ← 3 CUDA kernels
    ├── utils_gpu.h
    ├── utils_gpu.cu             ← host-side helpers, memory management
    ├── csv_io.h / csv_io.c      ← đọc/ghi CSV (host, thuần C)
    ├── test_dpc_aknn.cu         ← 7 unit tests
    ├── README.md
    └── outputs/
        ├── labels/
        ├── benchmarks/
        └── logs/
```

---

## SECTION D — FOLDER `original/` (Python / NumPy)

### D.1. Class Interface

```python
# config.py nằm cùng thư mục → import trực tiếp, không cần sys.path
from config import (
    LABELS_DIR, LOG_LEVEL, LOG_TO_FILE
)
import numpy as np

class DPCAKNN:
    """
    Thuật toán DPC-AKNN — cài đặt tham chiếu bằng Python / NumPy.

    Bài báo: "Density peak clustering based on nearest neighbors"
             Lin, Hou, Yuan — EAAI 160 (2025) 111981.

    Tham số:
        n_clusters (int): Số cluster n_c.
        k (int): Số láng giềng gần nhất. Khuyến nghị 10 ≤ k ≤ 30.
        random_state (int): Seed tái tạo kết quả.

    Thuộc tính sau fit():
        labels_  (np.ndarray): Nhãn cluster shape (n,). -1 = chưa phân bổ.
        centers_ (np.ndarray): Chỉ số tâm cluster shape (n_c,).
        rho_     (np.ndarray): Mật độ ρ_i shape (n,).
        delta_   (np.ndarray): Khoảng cách δ_i shape (n,).
        gamma_   (np.ndarray): Chỉ số γ_i shape (n,).
        d_c_     (float):      Ngưỡng khoảng cách d_c theo Eq. (7).
    """

    def fit(self, X: np.ndarray) -> "DPCAKNN": ...
    def fit_predict(self, X: np.ndarray) -> np.ndarray: ...
    def save_labels(self, filepath: str = None) -> None: ...
```

### D.2. Functions in `utils.py` (implement in this order)

```python
def compute_knn(X, k):
    """
    Tính kNN cho tất cả điểm dữ liệu dùng sklearn BallTree.
    Bài báo: nền tảng cho mọi bước.
    Song song hóa: Không.
    Trả về: (knn_indices, knn_distances) — shape (n,k) mỗi mảng.
    """

def compute_dc(knn_distances):
    """
    Tính d_c toàn cục: Eq. (6) → (8) → (9) → (7).
    Bài báo: Section 3.1.
    Song song hóa: Không.
    """

def compute_rho(knn_indices, knn_distances, d_c):
    """
    Tính mật độ ρ_i theo Eq. (5) — Gaussian kernel ràng buộc bởi kNN.
    Bài báo: Section 3.1, Eq. (5).
    Song song hóa: Không.
    """

def compute_delta(D, rho):
    """
    Tính δ_i theo Eq. (4). Cần ma trận khoảng cách đầy đủ D.
    Điểm có ρ cao nhất: δ = max(toàn bộ hàng i của D).
    Bài báo: Section 2.1, Eq. (4).
    Song song hóa: Không.
    """

def compute_gamma(rho, delta):
    """γ_i = ρ_i × δ_i. Bài báo: Section 3.2."""

def select_centers(gamma, n_clusters):
    """Chọn n_c điểm có γ lớn nhất. Bài báo: Alg. 1, Dòng 4."""

def compute_centroid(X, indices):
    """Tính centroid. Dùng cho điều kiện admission thứ 3."""

def check_admission(x_q_idx, x_p_idx, cluster_id, labels, X,
                    knn_indices, knn_distances, centroids, d_c):
    """
    3 điều kiện admission (Section 3.2.1):
      1) x_q chưa phân bổ
      2) d_pq ≤ mean(d_pt) với t ∈ N^k_{x_p}
      3) d(φ_j, x_q) ≤ d_c
    """

def build_initial_clusters(centers, knn_indices, knn_distances,
                            X, labels, n_clusters, d_c):
    """Xây dựng cluster ban đầu. Bài báo: Alg. 1, Dòng 5–15."""

def build_association_matrix(unassigned_indices, knn_indices,
                              labels, rho, n_clusters, D):
    """
    Tính A ∈ R^{n_u × n_c} theo Eq. (10).
    A(i,r) = Σ_{x_l ∈ kNN(x_i) ∩ C_r} (1/d_il) × ρ_l × ρ_i
    """

def run_association_loop(unassigned_indices, knn_indices,
                          labels, rho, n_clusters, D):
    """
    Vòng lặp phân bổ mật độ thấp, Eq. (10)+(11). Alg. 1, Dòng 17–22.
    Tuần tự — không thể song song hóa.
    """

def reallocate_by_voting(labels, knn_indices, rho, n_clusters, D):
    """Sửa lỗi bằng bỏ phiếu kNN. Alg. 1, Dòng 23."""

def allocate_remaining(labels, knn_indices, n_clusters, D):
    """Phân bổ điểm còn lại theo Eq. (12). Alg. 1, Dòng 24."""
```

### D.3. Tests and Demo

**`test_dpc_aknn.py`** — 7 unit tests (pytest):
```python
def test_eq6_per_point_dc()        # d_ci = d̄_i + σ_i
def test_eq7_global_dc()           # d_c = d̄_c + σ_c
def test_eq5_knn_gaussian_rho()    # ρ chỉ dùng kNN, không dùng toàn bộ data
def test_eq4_delta()               # δ với điểm có ρ cao nhất
def test_admission_conditions()    # 3 điều kiện admission riêng lẻ
def test_eq10_association_matrix() # công thức (1/d_il) × ρ_l × ρ_i
```

**`run_demo.py`**: Load data từ `DATA_DIR`, chạy `fit_predict`, in ACC/ARI/NMI,
lưu plot vào `PLOTS_DIR`, lưu nhãn vào `LABELS_DIR`.

`BENCHMARK_DIR`. Đây là baseline để so sánh với 2 phiên bản còn lại.

**`requirements.txt`**:
```
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
pytest>=7.0.0
```

---

## SECTION E — FOLDER `cpu_parallel/` (C99 + OpenMP)

### E.1. Parallelization Strategy

| Step | Complexity | OpenMP Strategy |
|------|-----------|-----------------|
| Full distance matrix D | O(n²d) | `#pragma omp parallel for schedule(static)` trên hàng i |
| Compute ρ_i (Eq. 5) | O(nk) | `#pragma omp parallel for` trên từng điểm i |
| Compute δ_i (Eq. 4) | O(n²) | `#pragma omp parallel for schedule(dynamic, CHUNK_SIZE)` |
| Build matrix A (Eq. 10) | O(n_u × n_c × k) | `#pragma omp parallel for` trên n_u điểm chưa phân bổ |
| kNN voting (Sec. 3.2.3) | O(n_c × n × k) | `#pragma omp parallel for` trên từng điểm |

**Assignment loop (Dòng 17–22)**: KHÔNG song song hóa — tuần tự theo thiết kế.

### E.2. Makefile

```makefile
# Makefile cho cpu_parallel/ — build với gcc + OpenMP

CC      = gcc
CFLAGS  = -std=c99 -O3 -march=native -Wall -Wextra
OMP     = -fopenmp
LIBS    = -lm

TARGET  = dpc_aknn_cpu
TEST    = test_dpc_aknn

SRCS    = main.c dpc_aknn.c utils_omp.c csv_io.c
TEST_SRCS = test_dpc_aknn.c dpc_aknn.c utils_omp.c csv_io.c

.PHONY: all test clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(OMP) -o $@ $^ $(LIBS)

test: $(TEST_SRCS)
	$(CC) $(CFLAGS) $(OMP) -o $(TEST) $^ $(LIBS)
	./$(TEST)

clean:
	rm -f $(TARGET) $(TEST) outputs/labels/*.csv outputs/benchmarks/*.csv
```

### E.3. Struct and Function Interface (`dpc_aknn.h`)

```c
#ifndef DPC_AKNN_H
#define DPC_AKNN_H

#include "config.h"

/*
 * DPCAKNN — Struct chứa toàn bộ trạng thái của thuật toán.
 *
 * Tương đương với class DPCAKNN trong Python (original/).
 * Tất cả mảng được cấp phát động; dùng dpcaknn_free() để giải phóng.
 */
typedef struct {
    /* Tham số thuật toán */
    int n_clusters;       /* Số cluster n_c */
    int k;                /* Số láng giềng gần nhất */
    int random_state;     /* Seed ngẫu nhiên */

    /* Kết quả sau khi fit() */
    int*    labels;       /* Nhãn cluster, shape (n,). -1 = chưa phân bổ */
    int*    centers;      /* Chỉ số tâm cluster, shape (n_c,) */
    real_t* rho;          /* Mật độ ρ_i, shape (n,) */
    real_t* delta;        /* Khoảng cách δ_i, shape (n,) */
    real_t* gamma;        /* Chỉ số γ_i, shape (n,) */
    real_t  d_c;          /* Ngưỡng khoảng cách toàn cục */

    /* Kích thước dữ liệu (lưu lại sau fit) */
    int n;
    int d;
} DPCAKNN;

/* Khởi tạo struct với tham số mặc định từ config.h */
DPCAKNN dpcaknn_init(int n_clusters, int k, int random_state);

/* Chạy toàn bộ Algorithm 1 trên X (shape: n × d, row-major) */
void dpcaknn_fit(DPCAKNN* model, const real_t* X, int n, int d);

/* fit() rồi trả về con trỏ labels (tiện dùng trong main) */
const int* dpcaknn_fit_predict(DPCAKNN* model, const real_t* X, int n, int d);

/* Lưu nhãn ra file CSV */
void dpcaknn_save_labels(const DPCAKNN* model, const char* filepath);

/* Giải phóng bộ nhớ động trong struct */
void dpcaknn_free(DPCAKNN* model);

#endif /* DPC_AKNN_H */
```

### E.4. Functions in `utils_omp.c` (implement in this order)

```c
/*
 * compute_knn_distances — Tính ma trận khoảng cách Euclid đầy đủ D[n×n].
 *
 * Mục đích: Tính D[i][j] = ||x_i − x_j||₂ cho mọi cặp (i, j).
 * Bài báo:  D dùng để tính δ_i (Eq. 4) và A(i,r) (Eq. 10).
 * Song song hóa: OpenMP parallel for, schedule(static), chia theo hàng i.
 *                Mỗi thread tính một dải hàng của D.
 *
 * Tham số:
 *   X:   Ma trận dữ liệu, shape (n, d), row-major, const real_t*.
 *   D:   Ma trận đầu ra, shape (n, n), row-major, real_t* (pre-allocated).
 *   n:   Số điểm.
 *   d:   Số chiều.
 */
void compute_distance_matrix(const real_t* X, real_t* D, int n, int d);

/*
 * find_knn — Tìm k láng giềng gần nhất cho từng điểm.
 *
 * Bài báo: Nền tảng cho mọi bước tính toán.
 * Song song hóa: OpenMP parallel for trên từng điểm i.
 *                Mỗi thread tìm kNN độc lập cho một điểm.
 *
 * Tham số:
 *   D:           Ma trận khoảng cách, shape (n, n), const real_t*.
 *   knn_indices: Chỉ số kNN đầu ra, shape (n, k), int* (pre-allocated).
 *   knn_dists:   Khoảng cách kNN đầu ra, shape (n, k), real_t*.
 *   n, k:        Kích thước.
 */
void find_knn(const real_t* D, int* knn_indices, real_t* knn_dists, int n, int k);

/*
 * compute_dc — Tính ngưỡng d_c toàn cục.
 *
 * Bài báo: Section 3.1, Eq. (6) → (8) → (9) → (7).
 * Song song hóa: OpenMP parallel for trên Bước 1 (tính d_ci cho từng điểm).
 *                reduction(+:sum) cho Bước 2 và 3.
 */
real_t compute_dc(const real_t* knn_dists, int n, int k);

/*
 * compute_rho — Tính mật độ ρ_i theo Eq. (5).
 *
 * Bài báo: Section 3.1, Eq. (5).
 * Song song hóa: #pragma omp parallel for trên từng điểm i.
 *   ρ_i = Σ_{j ∈ kNN(i)} exp(−d²_ij / d²_c)
 */
void compute_rho(const int* knn_indices, const real_t* knn_dists,
                 real_t d_c, real_t* rho, int n, int k);

/*
 * compute_delta — Tính δ_i theo Eq. (4).
 *
 * Bài báo: Section 2.1, Eq. (4).
 * Song song hóa: #pragma omp parallel for schedule(dynamic, OMP_CHUNK_SIZE).
 *   Dùng schedule(dynamic) vì mỗi điểm cần duyệt qua số điểm có ρ_j > ρ_i
 *   khác nhau — tải không đồng đều giữa các thread.
 */
void compute_delta(const real_t* D, const real_t* rho,
                   real_t* delta, int n);

/* compute_gamma, select_centers, compute_centroid: không song song (O(n) đơn giản) */
void compute_gamma(const real_t* rho, const real_t* delta,
                   real_t* gamma_arr, int n);
void select_centers(const real_t* gamma_arr, int n_clusters,
                    int* centers_out);
void compute_centroid(const real_t* X, const int* indices,
                      int n_indices, int d, real_t* centroid_out);

/*
 * check_admission — Kiểm tra 3 điều kiện admission (Section 3.2.1).
 * Không song song (gọi từ vòng lặp tuần tự).
 */
int check_admission(int x_q_idx, int x_p_idx, int cluster_id,
                    const int* labels, const real_t* X,
                    const int* knn_indices, const real_t* knn_dists,
                    const real_t* centroids, real_t d_c,
                    int n, int k, int d, int n_clusters);

/* build_initial_clusters — Alg. 1, Dòng 5–15. Không song song. */
void build_initial_clusters(const int* centers, const int* knn_indices,
                             const real_t* knn_dists, const real_t* X,
                             int* labels, int n_clusters,
                             real_t d_c, int n, int k, int d);

/*
 * build_association_matrix — Tính A ∈ R^{n_u × n_c} theo Eq. (10).
 *
 * Song song hóa: #pragma omp parallel for trên từng điểm chưa phân bổ i.
 *   Mỗi thread tính độc lập hàng A[i, :].
 */
void build_association_matrix(const int* unassigned, int n_u,
                               const int* knn_indices, const int* labels,
                               const real_t* rho, const real_t* D,
                               real_t* A, int n, int k, int n_clusters);

/* run_association_loop — Alg. 1, Dòng 17–22. Tuần tự, không song song. */
void run_association_loop(int* unassigned, int* n_u_ptr,
                           const int* knn_indices, int* labels,
                           const real_t* rho, const real_t* D,
                           int n, int k, int n_clusters);

/*
 * reallocate_by_voting — Sửa lỗi bằng bỏ phiếu kNN. Alg. 1, Dòng 23.
 *
 * Song song hóa: #pragma omp parallel for trên từng điểm.
 *   Mỗi thread tính phiếu bầu cho một điểm độc lập.
 *   Giai đoạn ghi nhãn cuối cùng thực hiện tuần tự để tránh race condition.
 */
void reallocate_by_voting(int* labels, const int* knn_indices,
                           const real_t* rho, const real_t* D,
                           int n, int k, int n_clusters);

/* allocate_remaining — Alg. 1, Dòng 24, Eq. (12). Song song trên từng điểm. */
void allocate_remaining(int* labels, const int* knn_indices,
                         const real_t* D, int n, int k, int n_clusters);
```

### E.5. Entry Point (`main.c`)

```c
/*
 * main.c — Entry point cho cpu_parallel.
 *
 * Sử dụng:
 *   ./dpc_aknn_cpu --input <file.csv>  (chạy với file CSV chỉ định)
 *
 * Đo thời gian bằng omp_get_wtime() — độ phân giải cao, tương thích OpenMP.
 * Ket qua nhan luu vao LABELS_DIR/cpu_labels_{timestamp}.csv
 */
```

### E.6. Tests (`test_dpc_aknn.c`)

7 unit tests tương tự Python, tự viết không dùng framework bên ngoài:
```c
static void test_eq6_per_point_dc(void);
static void test_eq7_global_dc(void);
static void test_eq5_knn_gaussian_rho(void);
static void test_eq4_delta(void);
static void test_admission_conditions(void);
static void test_eq10_association_matrix(void);

int main(void) {
    /* Chạy tất cả test, in PASS/FAIL, trả về exit code khác 0 nếu có lỗi */
}
```

---

## SECTION F — FOLDER `gpu_parallel/` (CUDA C)

### F.1. Parallelization Strategy

| Step | CUDA Strategy |
|------|---------------|
| Distance matrix D | Kernel 1: tiled 2D, shared memory |
| Density ρ (Eq. 5) | Kernel 2: 1D, 1 thread/point |
| Association matrix A (Eq. 10) | Kernel 3: 2D, 1 thread per (i, r) pair |
| Assignment loop (Alg. 1, Lines 17–22) | Host (sequential — cannot parallelize) |
| kNN voting correction | Kernel 4: 1D, 1 thread/point |

### F.2. Makefile

```makefile
# Makefile cho gpu_parallel/ — build với nvcc + CUDA

NVCC     = nvcc
NVFLAGS  = -O3 -arch=sm_75 -std=c++14 --compiler-options -Wall
LIBS     = -lm

TARGET   = dpc_aknn_gpu
TEST     = test_dpc_aknn_gpu

SRCS     = main.cu dpc_aknn.cu kernels.cu utils_gpu.cu csv_io.c
TEST_SRCS = test_dpc_aknn.cu kernels.cu utils_gpu.cu csv_io.c

.PHONY: all test clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

test: $(TEST_SRCS)
	$(NVCC) $(NVFLAGS) -o $(TEST) $^ $(LIBS)
	./$(TEST)

clean:
	rm -f $(TARGET) $(TEST) outputs/labels/*.csv outputs/benchmarks/*.csv
```

> Thay `-arch=sm_75` bằng compute capability phù hợp với GPU của bạn
> (sm_61 cho GTX 10xx, sm_86 cho RTX 30xx, sm_89 cho RTX 40xx).

### F.3. CUDA Kernel Specifications (`kernels.cu`)

#### Kernel 1 — Tiled Distance Matrix

```cuda
/*
 * pairwise_dist_tiled_kernel — Tính D[n×n] dùng tiled shared memory.
 *
 * Mục đích: Giảm số lần đọc global memory bằng cách load từng tile của X
 *           vào shared memory trước, sau đó tính khoảng cách trong shared mem.
 *
 * Song song hóa: CUDA kernel 2D
 *   Grid:  (ceil(n/TILE_SIZE), ceil(n/TILE_SIZE))
 *   Block: (TILE_SIZE, TILE_SIZE) — TILE_SIZE từ config.h
 *   Shared memory: 2 × TILE_SIZE × d × sizeof(float) bytes mỗi block
 *   Mỗi thread (i, j) tính D[i][j]
 *
 * Bài báo: D dùng cho Eq. (4) và Eq. (10).
 *
 * Tham số:
 *   X:  Ma trận dữ liệu đầu vào, shape (n, d), float* (device)
 *   D:  Ma trận khoảng cách đầu ra, shape (n, n), float* (device)
 *   n:  Số điểm
 *   d:  Số chiều
 */
__global__ void pairwise_dist_tiled_kernel(
    const float* __restrict__ X,
    float* D,
    int n, int d
);
```

#### Kernel 2 — Local Density ρ

```cuda
/*
 * compute_rho_kernel — Tính mật độ cục bộ ρ_i theo Eq. (5).
 *
 * Song song hóa: CUDA kernel 1D
 *   Grid:  (ceil(n / BLOCK_SIZE_1D),)  — BLOCK_SIZE_1D từ config.h
 *   Block: (BLOCK_SIZE_1D,)
 *   Mỗi thread xử lý 1 điểm i: duyệt k láng giềng, tính tổng Gaussian.
 *
 * Bài báo: Section 3.1, Eq. (5)
 *   ρ_i = Σ_{j ∈ kNN(i)} exp(−D[i][j]² / d_c²)
 *
 * Tham số:
 *   knn_indices: Chỉ số kNN, shape (n, k), int* (device)
 *   D:           Ma trận khoảng cách, shape (n, n), float* (device)
 *   rho:         Mật độ đầu ra, shape (n,), float* (device)
 *   d_c:         Ngưỡng khoảng cách (scalar), float
 *   n, k:        Kích thước
 */
__global__ void compute_rho_kernel(
    const int* __restrict__ knn_indices,
    const float* __restrict__ D,
    float* rho,
    float d_c,
    int n, int k
);
```

#### Kernel 3 — Association Matrix A

```cuda
/*
 * build_association_matrix_kernel — Tính A ∈ R^{n_u × n_c} theo Eq. (10).
 *
 * Song song hóa: CUDA kernel 2D
 *   Grid:  (ceil(n_u / BLOCK_SIZE_2D_X), ceil(n_c / BLOCK_SIZE_2D_Y))
 *   Block: (BLOCK_SIZE_2D_X, BLOCK_SIZE_2D_Y)
 *   Thread (i, r) tính A[i][r] cho điểm chưa phân bổ thứ i và cluster r.
 *   Mỗi thread duyệt k láng giềng của x_i, cộng dồn nếu x_l ∈ C_r.
 *
 * Bài báo: Section 3.2.2, Eq. (10)
 *   A(i, r) = Σ_{x_l ∈ kNN(x_i) ∩ C_r} (1/d_il) × ρ_l × ρ_i
 *
 * Tham số:
 *   knn_indices:        kNN của điểm chưa phân bổ, shape (n_u, k), int* (device)
 *   D_unassigned:       Khoảng cách tương ứng, shape (n_u, k), float* (device)
 *   cluster_labels:     Nhãn cluster hiện tại, shape (n,), int* (device). -1=chưa phân bổ
 *   rho:                Mật độ tất cả điểm, shape (n,), float* (device)
 *   A:                  Ma trận đầu ra, shape (n_u, n_c), float* (device)
 *   unassigned_global:  Chỉ số global của điểm chưa phân bổ, shape (n_u,), int* (device)
 *   n_u, n_c, k:        Kích thước
 */
__global__ void build_association_matrix_kernel(
    const int*   __restrict__ knn_indices,
    const float* __restrict__ D_unassigned,
    const int*   __restrict__ cluster_labels,
    const float* __restrict__ rho,
    float* A,
    const int* __restrict__ unassigned_global,
    int n_u, int n_c, int k
);
```

#### Kernel 4 — kNN Voting (Error Correction)

```cuda
/*
 * knn_voting_kernel — Bỏ phiếu kNN để sửa lỗi phân bổ. Alg. 1, Dòng 23.
 *
 * Song song hóa: CUDA kernel 1D
 *   Grid:  (ceil(n / BLOCK_SIZE_1D),)
 *   Block: (BLOCK_SIZE_1D,)
 *   Mỗi thread xử lý 1 điểm i: đếm phiếu cho từng cluster trong kNN,
 *   ghi kết quả vào new_labels[i].
 *   Sau khi kernel chạy xong, host copy new_labels → labels (tuần tự).
 *
 * Bài báo: Section 3.2.3, Dòng 23 của Algorithm 1.
 *
 * Tham số:
 *   knn_indices:  Chỉ số kNN, shape (n, k), int* (device)
 *   labels:       Nhãn cluster hiện tại, shape (n,), int* (device)
 *   rho:          Mật độ, shape (n,), float* (device)
 *   D:            Ma trận khoảng cách, shape (n, n), float* (device)
 *   new_labels:   Nhãn sau bỏ phiếu đầu ra, shape (n,), int* (device)
 *   n, k, n_c:    Kích thước
 */
__global__ void knn_voting_kernel(
    const int*   __restrict__ knn_indices,
    const int*   __restrict__ labels,
    const float* __restrict__ rho,
    const float* __restrict__ D,
    int* new_labels,
    int n, int k, int n_c
);
```

### F.4. Host Struct and Interface (`dpc_aknn.h` / `dpc_aknn.cu`)

```c
/*
 * DPCAKNN_GPU — Struct quản lý cả bộ nhớ host và device.
 *
 * Quy ước bộ nhớ:
 *   Tiền tố h_ = host (CPU RAM)
 *   Tiền tố d_ = device (GPU VRAM)
 */
typedef struct {
    /* Tham số */
    int n_clusters;
    int k;

    /* Host — kết quả cuối cùng */
    int*   h_labels;     /* shape (n,) */
    int*   h_centers;    /* shape (n_c,) */
    float* h_rho;        /* shape (n,) */
    float* h_delta;      /* shape (n,) */
    float* h_gamma;      /* shape (n,) */
    float  d_c;

    /* Device — bộ nhớ GPU còn giữ sau fit() để benchmark */
    float* d_X;          /* Ma trận dữ liệu, shape (n, d) */
    float* d_D;          /* Ma trận khoảng cách, shape (n, n) */
    float* d_rho;
    int*   d_labels;

    int n, d;
} DPCAKNN_GPU;

void dpcaknn_gpu_init(DPCAKNN_GPU* model, int n_clusters, int k);
void dpcaknn_gpu_fit(DPCAKNN_GPU* model, const float* h_X, int n, int d);
const int* dpcaknn_gpu_fit_predict(DPCAKNN_GPU* model, const float* h_X, int n, int d);
void dpcaknn_gpu_save_labels(const DPCAKNN_GPU* model, const char* filepath);
void dpcaknn_gpu_free(DPCAKNN_GPU* model);   /* giải phóng cả host và device */
```

### F.5. Memory Management (`utils_gpu.cu`)

```c
/*
 * gpu_malloc_check — Wrapper quanh cudaMalloc với kiểm tra lỗi.
 *
 * Mục đích: Đảm bảo mọi phép cấp phát GPU đều được kiểm tra,
 *           in thông báo lỗi tiếng Việt và thoát gracefully nếu thất bại.
 */
void* gpu_malloc_check(size_t bytes, const char* var_name);

/*
 * gpu_check_error — Kiểm tra lỗi CUDA sau mỗi kernel launch.
 *
 * Gọi ngay sau mỗi lệnh <<<...>>> để phát hiện lỗi launch hoặc runtime.
 * In file, dòng, và mô tả lỗi bằng tiếng Việt.
 */
void gpu_check_error(cudaError_t err, const char* file, int line);

/* Macro tiện dụng — bọc quanh mỗi kernel call */
#define CUDA_CHECK(call) gpu_check_error((call), __FILE__, __LINE__)

/*
 * gpu_batch_distance — Tính D theo batch để tránh OOM.
 *
 * Mục đích: Với dataset lớn, D đầy đủ có thể vượt VRAM.
 *           Hàm này tính từng batch GPU_BATCH_SIZE hàng một.
 * Bài báo:  D dùng cho Eq. (4) và Eq. (10).
 * Song song hóa: Mỗi batch gọi pairwise_dist_tiled_kernel.
 */
void gpu_batch_distance(const float* d_X, float* d_D, int n, int d);
```

---

## SECTION G — MANDATORY CODE STANDARDS

### G.1. Comment Rules

```c
/* ĐÚNG — Chú thích tiếng Việt, cụ thể, tham chiếu bài báo */
/* Tính khoảng cách trung bình từ x_i đến k láng giềng — dùng cho Eq. (6) */
d_bar_i = mean_distance(knn_dists + i * k, k);

/* SAI — tiếng Anh, không tham chiếu */
/* Calculate mean distance */
d_bar_i = mean_distance(knn_dists + i * k, k);
```

Trước mỗi khối logic lớn:
```c
/* === Bước 2: Tính ρ_i theo Eq. (5) — Gaussian kernel ràng buộc bởi kNN ===
 * Section 3.1, trang 4 của bài báo.
 * Song song hóa: #pragma omp parallel for trên từng điểm i.           */
#pragma omp parallel for schedule(static) num_threads(OMP_N_THREADS)
for (int i = 0; i < n; i++) { ... }
```

### G.2. Numerical Precision

- `original/`: `np.float64`
- `cpu_parallel/`: `double` (typedef `real_t` trong config.h)
- `gpu_parallel/`: `float` (typedef `real_t` trong config.h) — GPU tối ưu với float32
- Tránh chia cho 0: `d_ij` phải ≥ `1e-10` (hoặc `1e-7f` cho float)
- Điểm ρ cao nhất: `δ = max(toàn bộ hàng i của D)`, không phải `max(δ)`

### G.3. Timing

```c
/* cpu_parallel — dùng omp_get_wtime() */
#include <omp.h>
double t_start = omp_get_wtime();
dpcaknn_fit(&model, X, n, d);
double t_end = omp_get_wtime();
printf("Thời gian thực hiện: %.4f giây\n", t_end - t_start);

/* gpu_parallel — dùng CUDA Events */
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
dpcaknn_gpu_fit(&model, h_X, n, d);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("Thời gian thực hiện: %.4f giây\n", ms / 1000.0f);
```

### G.4. Saving Results

```c
/* Không hardcode đường dẫn — luôn dùng LABELS_DIR và BENCHMARK_DIR từ config.h */
char filepath[512];
snprintf(filepath, sizeof(filepath),
         LABELS_DIR "/labels_%s.csv", timestamp_str());
dpcaknn_save_labels(&model, filepath);
```

### G.5. Data Loading (`csv_io.c` — shared between cpu and gpu)

```c
/*
 * csv_read_matrix — Đọc file CSV thành ma trận float/double.
 *
 * Mục đích: Cung cấp cách đọc dữ liệu thống nhất cho cpu_parallel và
 *           gpu_parallel. File này viết bằng C thuần, không phụ thuộc
 *           thư viện ngoài.
 *
 * Tham số:
 *   filepath: Đường dẫn file CSV (chuỗi C).
 *   n_out:    Con trỏ nhận số hàng (số điểm).
 *   d_out:    Con trỏ nhận số cột (số chiều).
 * Trả về:
 *   float* được cấp phát động, row-major, shape (n, d).
 *   Người gọi chịu trách nhiệm free().
 */
float* csv_read_matrix(const char* filepath, int* n_out, int* d_out);

/*
 * csv_write_labels — Ghi nhãn cluster ra file CSV.
 */
void csv_write_labels(const char* filepath, const int* labels, int n);
```

---

## SECTION H — BUILD ORDER

### H.1. Build Sequence

1. Viết `theory/THEORY.md` (cơ sở lý thuyết)
2. Build **`original/`** (Python — baseline):
   - Viết `config.py`
   - Implement code → chạy 7 unit tests (pytest) → tất cả phải PASS
   - Chạy `benchmark.py` → ghi lại thời gian baseline
3. Build **`cpu_parallel/`** (C + OpenMP):
   - Viết `config.h` và `Makefile`
   - Implement code → `make test` → tất cả 7 tests PASS
   - Validate: ARI = 1.0 so với `original/` trên cùng file data từ `../data/`
4. Build **`gpu_parallel/`** (CUDA C):
   - Viết `config.h` và `Makefile`
   - Implement code → `make test` → tất cả 7 tests PASS
   - Validate: ARI = 1.0 so với `original/` trên cùng file data

### H.2. Implementation Order Within Each Folder

```
1. config.py / config.h
2. csv_io.c (nếu là C/CUDA) hoặc utils.py (Python)
3. utils_omp.c / utils_gpu.cu / kernels.cu (theo đúng thứ tự 11+ hàm ở trên)
4. dpc_aknn.c / dpc_aknn.cu (wire các hàm vào fit())
5. test_dpc_aknn (7 tests — phải PASS trước khi tiếp tục)
6. main.c / main.cu
7. README.md
8. Makefile / requirements.txt
```

---

## SECTION I — CRITICAL NOTES FOR THE AI

1. **Eq. (10) exactly**: `A(i,r) = Σ (1/d_il) × ρ_l × ρ_i` — has the extra `ρ_i` factor.

2. **Eq. (11) only updates column r\***: When assigning x_{i*} to C_{r*}, only update
   `A(m, r*)` for points x_m that have x_{i*} in their kNN. Do not touch other columns.

3. **Admission condition 2 uses the MEAN**: `d_pq ≤ (1/k) × Σ d_pt` — mean, not max.

4. **Admission condition 3 uses global d_c** (not per-point d_ci).

5. **Error correction (Section 3.2.3)**: Sort by density **descending**, process ALL points
   (including already-assigned ones) — this is *re-allocation*, not only unassigned points.
   In CUDA: collect votes per-thread, write new_labels to a separate buffer, then copy back.

6. **Eq. (12)**: `ζ_ir` = mean distance to k nearest neighbors **within cluster r** —
   not to all points of cluster r.

7. **Race conditions in OpenMP**: `reallocate_by_voting` reads and writes `labels[]`.
   Compute new label into a temporary array first; write back after the parallel section.

8. **GPU memory**: Always check for OOM. Use `GPU_BATCH_SIZE` from config.h when
   computing D for large datasets. Free device memory when no longer needed.

9. **Data shared, outputs separate**: All three implementations read from `../data/`.
   Never hardcode a path inside `data/`; always construct paths via `DATA_DIR`.

10. **Write `CHANGELOG.md`** at project root: list every simplification vs. the paper.
