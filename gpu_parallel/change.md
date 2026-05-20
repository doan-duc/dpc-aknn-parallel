# Changelog — DPC-AKNN GPU Parallel

## Phiên bản 3 (2026-05-16) — cuBLAS GEMM Approach (PLANNED)

### Tại sao v2 thất bại?

Shared memory tiling cho `rho[]` trong delta kernel KHÔNG giúp gì vì:
- `rho[j]` chỉ 4 bytes/điểm → đọc rất nhanh, không phải bottleneck
- Bottleneck thực sự là đọc `X[j]` = 784 float = **3136 bytes/điểm**
- Overhead của `__syncthreads()` và tile management thậm chí làm chậm thêm
- Kết quả: 143.9s (v2) vs 142.3s (v1) → **chậm hơn 1.6s**

### Phân tích sâu: Tại sao brute-force kernel chậm trên GPU?

**Vấn đề cốt lõi: Memory-Bound + Warp Divergence**

1. **Memory traffic**: Mỗi thread đọc X[j] (3136B) cho 70000 điểm j = 220 MB/thread.
   70000 thread × 220 MB = **15.3 TB** global memory reads.
   RTX 3060 bandwidth ~288 GB/s → **tối thiểu 53s** chỉ đọc dữ liệu.

2. **Warp Divergence từ Early Exit**: Khi một thread trong warp early-exit nhưng
   31 thread khác chưa → thread đó vẫn phải đợi. Trên GPU 32-thread warp,
   early exit hiệu quả thấp hơn nhiều so với CPU (mỗi thread chạy độc lập).

3. **L2 cache chỉ 3 MB** (RTX 3060 Laptop): X = 220 MB → cache hit rate cực thấp.
   So sánh: EPYC 9754 có **256 MB L3 cache** → toàn bộ X fit trong cache!

4. **Register pressure**: Mỗi thread cần 15 float best_dist + 15 int best_idx =
   120 bytes registers. Với 255 registers max/thread → chiếm 30 registers,
   giảm occupancy.

### Giải pháp: cuBLAS GEMM + Top-K Selection

**Ý tưởng cốt lõi**: Phân tách phép tính khoảng cách thành phép nhân ma trận:
```
||x_i - x_j||² = ||x_i||² + ||x_j||² - 2·(x_i · x_j)
                  ↑ norms    ↑ norms    ↑ GEMM (cực nhanh)
```

cuBLAS SGEMM tận dụng Tensor Cores, tiling tối ưu, shared memory pipeline →
đạt gần peak throughput (12.74 TFLOPS FP32 trên RTX 3060).

**Tính toán FLOP:**
- GEMM: n × d × n = 70000 × 784 × 70000 = **3.84 TFLOP**
- Tại 50% efficiency: 3.84 / (12.74 × 0.5) = **~0.6s** ← thay vì 108s!

**Tính toán VRAM** (RTX 3060 Laptop = 6 GB):
- d_X: 70000 × 784 × 4 = 220 MB
- d_norms: 70000 × 4 = 0.27 MB
- d_inner (batch): BS × 70000 × 4
  - BS=4000: 1.12 GB → tổng ~1.35 GB ← fits in 6 GB
  - BS=8000: 2.24 GB → tổng ~2.47 GB ← fits in 6 GB
- knn/rho/delta buffers: ~10 MB
→ **BS=4000 an toàn, 18 batches** (70000/4000 = 17.5)

### Luồng xử lý mới

```
Phase 1: kNN via GEMM (thay compute_knn_kernel)
├── Precompute norms[i] = ||X[i]||²                           ~0.001s
├── For each batch b (4000 rows):
│   ├── cublasSgemm: Inner[4000×70000] = -2 · X_batch · X^T   ~0.03s/batch
│   └── topk_kernel: D²=norms[i]+norms[j]+Inner → max-heap k  ~0.05s/batch
│   Total: 18 batches × 0.08s =                                ~1.5s
└── Copy knn_idx, knn_dist to host                             ~0.01s

Phase 2: d_c + rho (giữ nguyên, rất nhanh)                    ~0.002s

Phase 3: delta via GEMM (thay compute_delta_kernel)
├── For each batch b (4000 rows):
│   ├── cublasSgemm: Inner[4000×70000] (reuse buffer)          ~0.03s/batch
│   └── delta_kernel: D²=norms+Inner, find min where ρ_j>ρ_i  ~0.05s/batch
│   Total: 18 batches × 0.08s =                                ~1.5s
└── Copy delta to host                                         ~0.001s

Tổng ước tính Phase 1+2+3: ~3s (thay vì 136.8s → speedup 45×)
```

### Kernel mới cần viết

| Kernel | Mô tả |
|---|---|
| `compute_norms_kernel` | norms[i] = sum_p X[i*d+p]² — trivial, 1 thread/điểm |
| `topk_from_gemm_kernel` | Nhận Inner[BS×n] + norms → tính D² on-the-fly, duy trì max-heap k, ghi vào knn_idx/knn_dist. **Không cần lưu D²** — tính và dùng ngay |
| `delta_from_gemm_kernel` | Nhận Inner[BS×n] + norms + rho → tính D² on-the-fly, tìm min D² với ρ_j > ρ_i. **Không cần lưu D²** |

### Thay đổi build

```makefile
# Thêm -lcublas vào LIBS
LIBS = -lcublas
```

### Ước tính thời gian cuối cùng

| Bước | v1 (hiện tại) | v3 (GEMM) | Speedup |
|---|---|---|---|
| 1. kNN | 108.4s | ~1.5s | **72×** |
| 2. d_c | 0.002s | 0.002s | 1× |
| 3a. rho | 0.000s | 0.000s | 1× |
| 3b. delta | 28.4s | ~1.5s | **19×** |
| 4. centers | 0.008s | 0.008s | 1× |
| 5. BFS | 0.04s | 0.04s | 1× |
| 6. association | 5.3s | 5.3s | 1× |
| 7. voting | 0.01s | 0.01s | 1× |
| 8. outlier | 0.000s | 0.000s | 1× |
| **TỔNG** | **142.3s** | **~8.5s** | **~17×** |

So với CPU 256 threads (27s): GPU v3 dự kiến nhanh hơn **~3×**.

### Rủi ro và lưu ý

1. **Numerical precision**: cuBLAS GEMM tích lũy theo thứ tự khác → khoảng cách
   có thể lệch ~1e-6. Điều này có thể thay đổi danh sách kNN cho các điểm
   biên → ARI có thể thay đổi nhẹ (±0.01).

2. **VRAM 6GB**: BS=4000 an toàn. Nếu cần, có thể giảm xuống BS=2000 (36 batches,
   ~3s thay ~1.5s — vẫn rất nhanh).

3. **cuBLAS init overhead**: Lần gọi đầu tiên ~0.3-0.5s. Tạo handle sớm.

---

## Phiên bản 2 (2026-05-16) — Shared Memory Tiling (THẤT BẠI)

### Kernel thay đổi (`src/kernels.cu`)

| ID | Bước | Mô tả | Kết quả |
|---|---|---|---|
| OPT-1 | kNN (Bước 1) | Giữ nguyên — d=784 quá lớn để tile X vào shared memory | Không thay đổi |
| OPT-2 | Delta (Bước 3b) | Tiled shared memory cho rho[] (DELTA_TILE=256) | ❌ +1.6s (chậm hơn) |

**Lý do thất bại**: rho[] chỉ chiếm 4B/điểm, không phải bottleneck.
Overhead __syncthreads() mỗi tile > lợi ích từ shared memory cho rho.

### Logging & I/O (`src/main.cu`, `src/dpc_aknn.cu`, `src/utils_gpu.cu/.h`)

| ID | Mô tả |
|---|---|
| LOG-1 | Thêm hệ thống log kép `log_printf()` — in ra cả Console (stdout) lẫn file trong `output/logs/` |
| LOG-2 | In tiến độ chi tiết 8 bước kèm thời gian (`[DPC-AKNN] Buoc X/8: ... -> Xong. (%.3f s)`) |
| LOG-3 | In bảng tổng kết cuối cùng giống hệt bản CPU |
| LOG-4 | Tự động tạo file log `output/logs/gpu_run_<timestamp>.txt` |

### Chỉ số đánh giá (`src/utils_gpu.cu/.h`)

| ID | Mô tả |
|---|---|
| METRIC-1 | Implement `adjusted_rand_index()` — ARI chuẩn với safety check cho nhãn âm |
| METRIC-2 | Implement `normalized_mutual_info()` — NMI chuẩn (Entropy + MI) |
| METRIC-3 | Implement `clustering_accuracy()` — ACC (Greedy Max-Weight Matching) |
| METRIC-4 | Xóa bỏ placeholder `(est)`, tất cả chỉ số tính toán thực tế |

### Đọc dữ liệu & Tiện ích

| ID | Mô tả |
|---|---|
| IO-1 | `csv_read_labels()` — đọc nhãn gốc, auto-detect header |
| IO-2 | Fix bug nhãn bị lệch 1 vị trí khi file không có header |
| IO-3 | `get_time_sec()` — QueryPerformanceCounter (Windows) |

### CLI (`src/main.cu`)

| ID | Mô tả |
|---|---|
| CLI-1 | Hỗ trợ `--input`, `--labels`, `--clusters`, `--k` |

---

## Phiên bản 1 (2026-05-14) — Bản CUDA C ban đầu

### Kiến trúc

| File | Mô tả |
|---|---|
| `src/config.h` | Cấu hình: GPU device, block size, tile size, max K, epsilon |
| `src/kernels.cu/.cuh` | CUDA kernels cho 8 bước DPC-AKNN |
| `src/dpc_aknn.cu/.h` | Host orchestration — điều phối kernel + quản lý bộ nhớ |
| `src/utils_gpu.cu/.h` | Tiện ích: CUDA error check, CSV I/O |
| `src/main.cu` | Entry point |
| `Makefile` | Build: nvcc + cl.exe (MSVC) |

### Tối ưu có từ v1

- **Không cấp phát D[n×n]** — kNN trực tiếp, chỉ giữ knn_idx/dist[n×k]
- **Max-heap O(n log k)** cho top-k selection
- **Early exit** trên dist² khi vượt ngưỡng
- **dist² thay dist** — sqrt() chỉ gọi k lần cuối
- **Incremental centroid O(d)** trong BFS
- **Reverse-kNN** cho association update O(nk)
- **BFS serial** giữ thứ tự claim nhãn

---

## Benchmark (Fashion-MNIST: n=70000, d=784, k=15, clusters=10)

| Phiên bản | Thời gian | ARI | NMI | ACC |
|---|---|---|---|---|
| CPU 16 threads (local) | 1304.6s | 0.3764 | 0.5711 | 0.5240 |
| CPU 256 threads (EPYC 9754) | **27s** | 0.3764 | 0.5711 | 0.5240 |
| GPU v1 (RTX 3060 Laptop 6GB) | 142.3s | 0.4070 | 0.5771 | 0.5237 |
| GPU v2 (RTX 3060 Laptop 6GB) | 143.9s | 0.4070 | 0.5771 | 0.5237 |
| GPU v3 (RTX 3060 Laptop 6GB) | *ước tính ~8.5s* | ~0.40 | ~0.57 | ~0.52 |
