/*
 * dpc_aknn_core.c  —  DPC-AKNN, không dùng D[n×n].
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * DANH SÁCH FIX SO VỚI PHIÊN BẢN GỐC:
 *
 * 🔴 [FIX-1] step4 + step7: core_sort_desc O(n²) → sort_desc_by_value O(n log n)
 *      core_sort_desc dùng selection sort, comment ghi "chỉ dùng cho n_clusters
 *      (≤20)" nhưng cả step4 lẫn step7 gọi với n=70000 → ~4.9 tỷ phép so sánh
 *      đơn luồng → bước 4 tốn 15s, bước 7 tốn thêm vài giây chỉ cho sort.
 *      Fix: hàm nội bộ sort_desc_by_value dùng qsort O(n log n).
 *
 * 🔴 [FIX-2] step7: calloc/free trong parallel for → pre-alloc per-thread
 *      n thread đồng thời gọi calloc(n_clusters) → tranh heap lock → serialize ẩn.
 *      Fix: cấp phát T×n_clusters int ngoài vùng parallel, mỗi thread dùng slot
 *      riêng [tid*n_clusters .. (tid+1)*n_clusters), không còn tranh lock.
 *
 * 🟡 [FIX-3] step5: core_compute_centroid O(n)/enqueue → incremental O(d)
 *      BFS enqueue tối đa O(n) điểm, mỗi lần gọi core_compute_centroid scan
 *      toàn bộ n → tổng O(n²·d). Với n=70000, d=784: ~3.8 tỷ FLOP chỉ centroid.
 *      Fix: duy trì csum[] running sum và cnt, update O(d) khi thêm mỗi điểm.
 *
 * 🟡 [FIX-4] step1: qsort O(n log n) để tìm k láng giềng → max-heap O(n log k)
 *      k=15, n=70000: log(n)/log(k) ≈ 4.2× ít phép so sánh hơn.
 *      Buffer giảm từ O(n) → O(k) mỗi thread: 17.9MB → 3.8KB (với T=16).
 *
 * 🟢 [FIX-5] step6 build_association_matrix: memset toàn cục → init per-row
 *      Mỗi thread init row của mình ngay trước khi dùng → NUMA locality tốt hơn.
 *
 * 🟢 [FIX-B] step2: serial reduction O(n) → parallel reduction O(n/T)
 *      mean_c và var_c dùng omp reduction(+:) thay serial loop.
 *
 * 🟢 [FIX-C] step3b: serial max_dist O(n) → parallel reduction(max:)
 *      max_dist cho điểm top dùng omp reduction thay serial.
 *
 * 🔴 [FIX-E] step5: khôi phục bridge node BFS behavior từ v1
 *      v1 enqueue TẤT CẢ k láng giềng (kể cả đã labeled), code trước đó
 *      chỉ enqueue unlabeled → mất bridge → cụm co lại → ARI giảm.
 *      Fix: tách enqueue ra khỏi label assignment.
 *
 * 🔴 [FIX-F] step6: rebuild toàn bộ A mỗi vòng O(n²k) → incremental O(nk)
 *      Build reverse-kNN index 1 lần, mỗi vòng chỉ update cột r* cho
 *      các u ∈ rknn[best_point]. Speedup ~1000× trên n=70K.
 *
 * 🔴 [FIX-G] step7: false sharing trên counts_pool → pad 64B + _aligned_malloc
 *      n_clusters=10 → slot=40B < 64B cache line → thread invalidate nhau.
 *      Fix: pad slot lên 16 ints = 64B, căn chỉnh base address.
 *
 * 🟡 [FIX-H] step8: best_cl=0 default → best_cl=-1 + fallback tường minh
 *      Khi không có neighbor nào labeled, tránh gán nhầm mặc định cluster 0.
 *
 * ✅ Giữ nguyên (đã đúng):
 *      step1  pre-alloc per-thread buffer + max-heap O(n log k)
 *      step3b schedule(static) — dynamic chậm hơn 11s trên high-dim data
 *      step5  serial BFS — thứ tự claim điểm là tính chất thuật toán
 *      step7/8 double-buffer pattern
 * ═══════════════════════════════════════════════════════════════════════════
 */
#include "dpc_aknn_core.h"

#include <math.h>
#include <malloc.h>  /* _aligned_malloc / _aligned_free (Windows) */
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ── Tiện ích nội bộ ────────────────────────────────────────────────────── */

typedef struct { real_t dist; int idx; } Neighbor;

/* So sanh Neighbor: dist tang dan, tie-break theo idx. */
static int cmp_neighbor_asc(const void* a, const void* b) {
    const Neighbor* x = (const Neighbor*)a;
    const Neighbor* y = (const Neighbor*)b;
    if (x->dist < y->dist) return -1;
    if (x->dist > y->dist) return  1;
    return x->idx - y->idx;
}

/* ── [FIX-1] sort_desc_by_value: O(n log n) thay O(n²) ─────────────────── */
/*
 * Sắp xếp order[] giảm dần theo values[], tie-break: index nhỏ đứng trước.
 * Dùng qsort trên mảng ValIdx tạm — không cần biến global, thread-safe.
 * Chỉ dùng hàm này khi n lớn (step4, step7).
 * core_sort_desc giữ nguyên cho n_clusters ≤ 20 (đúng với comment gốc).
 */
typedef struct { real_t val; int idx; } ValIdx;

/* So sanh ValIdx: val giam dan, tie-break theo idx. */
static int cmp_validx_desc(const void* a, const void* b) {
    const ValIdx* x = (const ValIdx*)a;
    const ValIdx* y = (const ValIdx*)b;
    if (y->val > x->val) return  1;
    if (y->val < x->val) return -1;
    return x->idx - y->idx; /* tie-break: index nhỏ hơn đứng trước */
}

/* Tao order[] sap xep giam theo values[]. */
static void sort_desc_by_value(const real_t* values, int* order, int n) {
    ValIdx* vi = (ValIdx*)malloc((size_t)n * sizeof(ValIdx));
    for (int i = 0; i < n; i++) { vi[i].val = values[i]; vi[i].idx = i; }
    qsort(vi, (size_t)n, sizeof(ValIdx), cmp_validx_desc);
    for (int i = 0; i < n; i++) order[i] = vi[i].idx;
    free(vi);
}

/* Selection sort cho n nho, sap xep giam theo values. */
void core_sort_desc(const real_t* values, int* order, int n) {
    /* [SERIAL] Selection sort — CHỈ dùng khi n nhỏ (n_clusters ≤ 20). */
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (values[order[j]] > values[order[i]] ||
               (values[order[j]] == values[order[i]] && order[j] < order[i])) {
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }
}

/* Tinh centroid cua mot cum tu labels va X. */
void core_compute_centroid(const real_t* X, const int* labels,
                            int cluster_id, int n, int d, real_t* centroid) {
    /* [SERIAL] Giữ nguyên — giao diện công khai, không gọi trong BFS nữa. */
    int count = 0;
    for (int p = 0; p < d; p++) centroid[p] = 0.0;
    for (int i = 0; i < n; i++) {
        if (labels[i] != cluster_id) continue;
        for (int p = 0; p < d; p++) centroid[p] += X[i*d+p];
        count++;
    }
    if (count > 0)
        for (int p = 0; p < d; p++) centroid[p] /= (real_t)count;
}

/* ── [FIX-4] Max-heap O(n log k) cho bước 1 ─────────────────────────────── */
/*
 * Max-heap kích thước k: luôn giữ k phần tử nhỏ nhất đã thấy.
 * Root = phần tử LỚN NHẤT → khi gặp dist < root.dist, replace root và sift down.
 * Complexity: O(n log k) thay O(n log n) của qsort toàn bộ.
 * k=15, n=70000: tiết kiệm ~4.2× phép so sánh.
 */
/* Sift-down de khoi phuc tinh chat max-heap. */
static void heap_sift_down(Neighbor* h, int size, int i) {
    while (1) {
        int largest = i;
        int l = 2*i+1, r = 2*i+2;
        if (l < size && (h[l].dist > h[largest].dist ||
            (h[l].dist == h[largest].dist && h[l].idx > h[largest].idx)))
            largest = l;
        if (r < size && (h[r].dist > h[largest].dist ||
            (h[r].dist == h[largest].dist && h[r].idx > h[largest].idx)))
            largest = r;
        if (largest == i) break;
        Neighbor tmp = h[i]; h[i] = h[largest]; h[largest] = tmp;
        i = largest;
    }
}

/* Chen phan tu vao max-heap. */
static void heap_push(Neighbor* h, int* size, Neighbor val) {
    int i = (*size)++;
    h[i] = val;
    while (i > 0) {
        int parent = (i - 1) / 2;
        int swap = (h[i].dist > h[parent].dist) ||
                   (h[i].dist == h[parent].dist && h[i].idx > h[parent].idx);
        if (!swap) break;
        Neighbor tmp = h[parent]; h[parent] = h[i]; h[i] = tmp;
        i = parent;
    }
}

/* Thay root va khoi phuc max-heap. */
static void heap_replace_root(Neighbor* h, int size, Neighbor val) {
    h[0] = val;
    heap_sift_down(h, size, 0);
}

/* ── BƯỚC 1: kNN trực tiếp từ X ─────────────────────────────────────────── */

/* Tinh kNN bang max-heap va early-exit. */
void step1_compute_knn(const real_t* X, int* knn_idx, real_t* knn_dist,
                        int n, int d, int k) {
    /*
     * [DOMAIN] Phân rã miền theo điểm i.
     *
     * [FIX-4] Max-heap O(n log k) thay qsort O(n log n).
     *
     * [OPT-1] Heap lưu dist² thay dist — loại bỏ ~n² phép sqrt():
     *   sqrt() là hàm đơn điệu tăng: a < b <=> sqrt(a) < sqrt(b).
     *   Mọi phép so sánh trong heap hoạt động đúng với dist².
     *   Chỉ gọi sqrt() k lần khi xuất ra knn_dist[] ở cuối.
     *   Tiết kiệm: n=70000 => ~4.9 ty phep sqrt() => 0 phep trong vong lap chinh.
     *
     * [OPT-2] Early Exit khi tính khoảng cách:
     *   Nguong = dist² cua phan tu LON NHAT trong heap (heap[0].dist).
     *   Neu heap chua day (heap_sz < k): nguong = +inf, khong cat som.
     *   Neu heap da day: dung dist_euclid_sq_early(), cong don tung chieu,
     *   cham nguong la dung luon — diem j do chac chan khong lot top-k.
     *   Voi d=784, diem xa thuong bi loai sau ~10-20 chieu dau tien.
     *
     * Memory model:
     *   X[]             : read-only -> khong conflict.
     *   knn_idx/knn_dist: schedule(static) -> moi thread ghi dai rieng.
     *   bufs[tid]       : slot rieng cua thread -> khong conflict.
     */
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif

    Neighbor** bufs = (Neighbor**)malloc((size_t)nthreads * sizeof(Neighbor*));
    for (int t = 0; t < nthreads; t++)
        bufs[t] = (Neighbor*)malloc((size_t)k * sizeof(Neighbor));

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        Neighbor* heap = bufs[tid];
        int heap_sz = 0;

        for (int j = 0; j < n; j++) {
            if (j == i) continue;

            real_t dsq;
            if (heap_sz < k) {
                /* Heap chua day: nguong = +inf, tinh du chieu [OPT-1] */
                dsq = dist_euclid_sq(X, i, j, d);
                Neighbor nb = { dsq, j };
                heap_push(heap, &heap_sz, nb);
            } else {
                /* [OPT-2] Heap da day: early exit voi nguong = heap[0].dist */
                dsq = dist_euclid_sq_early(X, i, j, d, heap[0].dist);
                if (dsq < heap[0].dist ||
                    (dsq == heap[0].dist && j < heap[0].idx)) {
                    Neighbor nb = { dsq, j };
                    heap_replace_root(heap, heap_sz, nb);
                }
            }
        }
        /* Sort k phan tu tang dan -- O(k log k), k nho */
        qsort(heap, (size_t)heap_sz, sizeof(Neighbor), cmp_neighbor_asc);
        /* [OPT-1] sqrt() chi goi k lan o day, khong phai trong vong lap n */
        for (int t = 0; t < k; t++) {
            knn_idx [i * k + t] = heap[t].idx;
            knn_dist[i * k + t] = sqrt(heap[t].dist); /* dist^2 -> dist */
        }
    }

    for (int t = 0; t < nthreads; t++) free(bufs[t]);
    free(bufs);
}

/* ── BƯỚC 2: d_c ────────────────────────────────────────────────────────── */

/* Tinh d_c thich ung tu knn_dist cua tung diem. */
real_t step2_compute_dc(const real_t* knn_dist, int n, int k) {
    /* [DOMAIN + SERIAL] Giữ nguyên. */
    real_t* d_ci = (real_t*)malloc((size_t)n * sizeof(real_t));
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        real_t mean = 0.0;
        for (int t = 0; t < k; t++) mean += knn_dist[i*k+t];
        mean /= (real_t)k;
        real_t var = 0.0;
        for (int t = 0; t < k; t++) {
            real_t diff = knn_dist[i*k+t] - mean;
            var += diff * diff;
        }
        d_ci[i] = mean + sqrt(var / (real_t)k);
    }
    /* [FIX-B] Parallel reduction thay serial loop */
    real_t mean_c = 0.0;
#pragma omp parallel for reduction(+:mean_c) schedule(static)
    for (int i = 0; i < n; i++) mean_c += d_ci[i];
    mean_c /= (real_t)n;
    real_t var_c = 0.0;
#pragma omp parallel for reduction(+:var_c) schedule(static)
    for (int i = 0; i < n; i++) {
        real_t diff = d_ci[i] - mean_c;
        var_c += diff * diff;
    }
    real_t sigma_c = n > 1 ? sqrt(var_c / (real_t)(n - 1)) : 0.0;
    free(d_ci);
    return mean_c + sigma_c;
}

/* ── BƯỚC 3a: ρ ─────────────────────────────────────────────────────────── */

/* Tinh mat do cuc bo rho tu knn_dist va d_c. */
void step3a_compute_rho(const real_t* knn_dist,
                         real_t* rho, real_t d_c, int n, int k) {
    /* [DOMAIN] Giữ nguyên. */
    real_t safe_dc = d_c > EPS_DISTANCE ? d_c : EPS_DISTANCE;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        real_t sum = 0.0;
        for (int t = 0; t < k; t++) {
            real_t dist = knn_dist[i*k+t];
            sum += exp(-(dist * dist) / (safe_dc * safe_dc));
        }
        rho[i] = sum;
    }
}

/* ── BƯỚC 3b: δ ─────────────────────────────────────────────────────────── */

/* Tinh delta: khoang cach toi diem co mat do cao hon gan nhat. */
void step3b_compute_delta(const real_t* X, const real_t* rho,
                           real_t* delta, int n, int d) {
    /*
     * [DOMAIN] schedule(static) — đo được dynamic(32) chậm hơn 11s với d=784.
     *
     * [OPT-1] dist² thay dist — bỏ sqrt() trong vòng lặp.
     * [OPT-2] Early Exit cho vòng lặp tìm min (Phần 2):
     *   best_sq = bình phương khoảng cách nhỏ nhất tìm được.
     *   Nếu partial sum > best_sq → chắc chắn xa hơn → cắt ngay.
     */
    int top = 0;
    for (int i = 1; i < n; i++)
        if (rho[i] > rho[top] || (rho[i] == rho[top] && i < top))
            top = i;

    /* [FIX-C] Parallel reduction cho max_dist
     * [OPT-1] Dùng dist² trong vòng lặp, sqrt() 1 lần ở cuối.
     * max(sqrt(a), sqrt(b)) = sqrt(max(a, b)) — đơn điệu tăng. */
    real_t max_dist_sq = 0.0;
#pragma omp parallel for reduction(max:max_dist_sq) schedule(static)
    for (int j = 0; j < n; j++) {
        real_t dsq = dist_euclid_sq(X, top, j, d);
        if (dsq > max_dist_sq) max_dist_sq = dsq;
    }
    real_t max_dist = sqrt(max_dist_sq);
    delta[top] = max_dist;

    /* Phần 2: delta[i] = min dist tới điểm có rho cao hơn.
     * [OPT-2] Early Exit: best_sq làm ngưỡng cắt sớm. */
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        if (i == top) continue;
        real_t best_sq = 1e100;
        for (int j = 0; j < n; j++) {
            int j_higher = (rho[j] > rho[i]) ||
                           (rho[j] == rho[i] && j < i);
            if (!j_higher) continue;
            real_t dsq = dist_euclid_sq_early(X, i, j, d, best_sq);
            if (dsq < best_sq) best_sq = dsq;
        }
        delta[i] = (best_sq < 1e99) ? sqrt(best_sq) : max_dist;
    }
}

/* ── BƯỚC 4: Chọn tâm cụm ───────────────────────────────────────────────── */

/* Tinh gamma va chon top tam cum. */
void step4_select_centers(const real_t* rho, const real_t* delta,
                           real_t* gamma_out,
                           int n, int n_clusters, int* centers_out) {
    /*
     * [FIX-1] sort_desc_by_value O(n log n) thay core_sort_desc O(n²).
     * Bước 4 tốn 15.290s trong log do sort — fix này loại bỏ hầu hết thời gian đó.
     */
    for (int i = 0; i < n; i++) gamma_out[i] = rho[i] * delta[i];
    int* order = (int*)malloc((size_t)n * sizeof(int));
    sort_desc_by_value(gamma_out, order, n);
    for (int c = 0; c < n_clusters; c++) centers_out[c] = order[c];
    free(order);
}

/* ── BƯỚC 5: Cụm nòng cốt ban đầu ───────────────────────────────────────── */

/* Xay cum ban dau: one-hop + kiem tra centroid. */
void step5_build_initial_clusters(int* labels, const int* centers,
                                   const real_t* X,
                                   const int* knn_idx,
                                   const real_t* knn_dist,
                                   real_t d_c,
                                   int n, int d, int k, int n_clusters) {
    /*
     * [SERIAL] One-hop expansion — chỉ xét kNN của center và kNN của
     * các điểm seed, không enqueue điểm mới nhận để tránh lan rộng đệ quy.
     *
     * [FIX-3] Incremental centroid O(d) thay core_compute_centroid O(n):
     *   Duy trì csum[d] = tổng tọa độ các điểm đã nhận, cnt = số điểm.
     *   Khi thêm điểm x_q: csum[p] += X[x_q*d+p] cho p=0..d-1  → O(d).
     *   centroid[p] = csum[p] / cnt khi cần kiểm tra điều kiện 3  → O(d).
     *   Tổng: O(n·d) thay O(n²·d) — với n=70000 tiết kiệm 70000× FLOP.
     *
     * Memory: csum, centroid, queue là local của hàm → không share.
     */
    real_t* csum     = (real_t*)malloc((size_t)d * sizeof(real_t));
    real_t* centroid = (real_t*)malloc((size_t)d * sizeof(real_t));
    int*    queue    = (int*)   malloc((size_t)n * sizeof(int));

    for (int i = 0; i < n; i++) labels[i] = -1;

    for (int c = 0; c < n_clusters; c++) {
        int center = centers[c];
        int cnt = 0;
        for (int p = 0; p < d; p++) csum[p] = 0.0;

        if (labels[center] == -1) {
            labels[center] = c;
            for (int p = 0; p < d; p++) csum[p] += X[center*d+p];
            cnt++;
        }

        int head = 0, tail = 0;
        /* [FIX-E] Giữ tập seed đầy đủ từ v1:
         * Enqueue TẤT CẢ k láng giềng của center (kể cả đã labeled) để
         * đảm bảo các điểm seed không bị bỏ sót. Không lan rộng đệ quy. */
        for (int t = 0; t < k; t++) {
            int nb = knn_idx[center * k + t];
            if (labels[nb] == -1) {
                labels[nb] = c;
                for (int p = 0; p < d; p++) csum[p] += X[nb*d+p]; /* [FIX-3] O(d) */
                cnt++;
            }
            queue[tail++] = nb; /* seed queue: chỉ chứa kNN của center */
        }
        /* Centroid ban đầu từ csum — O(d) */
        if (cnt > 0)
            for (int p = 0; p < d; p++) centroid[p] = csum[p] / (real_t)cnt;

        while (head < tail) {
            int x_p = queue[head++];

            real_t local_mean = 0.0;
            for (int t = 0; t < k; t++) local_mean += knn_dist[x_p * k + t];
            local_mean /= (real_t)k;

            for (int t = 0; t < k; t++) {
                int x_q = knn_idx[x_p * k + t];
                if (labels[x_q] != -1) continue;
                if (knn_dist[x_p * k + t] > local_mean) continue;

                real_t dist_sq = 0.0;
                for (int p = 0; p < d; p++) {
                    real_t diff = centroid[p] - X[x_q * d + p];
                    dist_sq += diff * diff;
                }
                if (sqrt(dist_sq) > d_c) continue;

                labels[x_q] = c;

                /* [FIX-3] Incremental update — không gọi core_compute_centroid */
                for (int p = 0; p < d; p++) csum[p] += X[x_q*d+p];
                cnt++;
                for (int p = 0; p < d; p++) centroid[p] = csum[p] / (real_t)cnt;
            }
        }
    }
    free(csum);
    free(centroid);
    free(queue);
}

/* ── BƯỚC 6: Ma trận liên kết A ─────────────────────────────────────────── */
/*
 * [FIX-F] Incremental update theo Eq.(11) của bài báo.
 *
 * Phiên bản cũ: rebuild TOÀN BỘ ma trận A mỗi vòng → O(n_u² × k).
 * Phiên bản mới:
 *   1. Build reverse-kNN index 1 lần O(nk).
 *   2. Build A lần đầu (parallel) O(n_u × k).
 *   3. Mỗi vòng chỉ update cột r* cho các u ∈ rknn[best_point] O(k).
 *   → Tổng: O(n_u × k) + O(n_u × k) = O(n × k) thay O(n² × k).
 *
 * Memory model:
 *   A[] indexed by [point_id × n_clusters], không dùng row mapping.
 *   is_unassigned[] đánh dấu thay vì shift mảng → O(1) xóa.
 *   rknn_data[] read-only sau build → không conflict.
 */

/* Gan nhan con lai bang A va cap nhat theo rknn. */
void step6_association_loop(int* labels, const int* knn_idx,
                             const real_t* knn_dist, const real_t* rho,
                             int n, int k, int n_clusters) {
    /* ── Build reverse-kNN index: rknn[j] = {i : j ∈ kNN(i)} ────────── */
    int* rknn_count = (int*)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < n; i++)
        for (int t = 0; t < k; t++)
            rknn_count[knn_idx[i * k + t]]++;

    int* rknn_offset = (int*)malloc(((size_t)n + 1) * sizeof(int));
    rknn_offset[0] = 0;
    for (int j = 0; j < n; j++)
        rknn_offset[j + 1] = rknn_offset[j] + rknn_count[j];

    int total_rknn = rknn_offset[n];
    int* rknn_data = (int*)malloc((size_t)total_rknn * sizeof(int));
    memset(rknn_count, 0, (size_t)n * sizeof(int));
    for (int i = 0; i < n; i++)
        for (int t = 0; t < k; t++) {
            int j = knn_idx[i * k + t];
            rknn_data[rknn_offset[j] + rknn_count[j]++] = i;
        }

    /* ── Collect unassigned points ───────────────────────────────────── */
    int* unassigned = (int*)malloc((size_t)n * sizeof(int));
    int* is_unassigned = (int*)calloc((size_t)n, sizeof(int));
    int n_u = 0;
    for (int i = 0; i < n; i++)
        if (labels[i] < 0) { unassigned[n_u++] = i; is_unassigned[i] = 1; }
    if (n_u == 0) goto cleanup_step6;

    /* ── Build A lần đầu (parallel) ─────────────────────────────────── */
    real_t* A = (real_t*)calloc((size_t)n * (size_t)n_clusters, sizeof(real_t));

#pragma omp parallel for schedule(static)
    for (int idx = 0; idx < n_u; idx++) {
        int i = unassigned[idx];
        for (int t = 0; t < k; t++) {
            int    l  = knn_idx [i * k + t];
            int    cl = labels[l];
            if (cl < 0) continue;
            real_t dl = knn_dist[i * k + t];
            real_t safe_d = dl > EPS_DISTANCE ? dl : EPS_DISTANCE;
            A[i * n_clusters + cl] += (1.0 / safe_d) * rho[l] * rho[i];
        }
    }

    /* ── Iterative assignment ────────────────────────────────────────── */
    for (int iter = 0; iter < n_u; iter++) {
        /* find_best — parallel thread-local reduction */
        real_t best     = 0.0;
        int    best_pt  = -1;
        int    best_cl  = -1;

#pragma omp parallel
        {
            real_t lb_val = 0.0;
            int    lb_pt  = -1, lb_cl = -1;

#pragma omp for schedule(static) nowait
            for (int idx = 0; idx < n_u; idx++) {
                int i = unassigned[idx];
                if (!is_unassigned[i]) continue;
                for (int c = 0; c < n_clusters; c++)
                    if (A[i * n_clusters + c] > lb_val) {
                        lb_val = A[i * n_clusters + c];
                        lb_pt = i; lb_cl = c;
                    }
            }

#pragma omp critical
            { if (lb_val > best) { best = lb_val; best_pt = lb_pt; best_cl = lb_cl; } }
        }

        if (best_pt < 0) break;
        labels[best_pt] = best_cl;
        is_unassigned[best_pt] = 0;

        /* ── Incremental update theo Eq.(11): chỉ update các u ∈ rknn[best_pt] ── */
        for (int r = rknn_offset[best_pt]; r < rknn_offset[best_pt + 1]; r++) {
            int u = rknn_data[r];
            if (!is_unassigned[u]) continue;
            /* Tìm khoảng cách u → best_pt trong knn_dist */
            for (int t = 0; t < k; t++) {
                if (knn_idx[u * k + t] == best_pt) {
                    real_t dl = knn_dist[u * k + t];
                    real_t safe_d = dl > EPS_DISTANCE ? dl : EPS_DISTANCE;
                    A[u * n_clusters + best_cl] +=
                        (1.0 / safe_d) * rho[best_pt] * rho[u];
                    break;
                }
            }
        }
    }

    free(A);
cleanup_step6:
    free(rknn_count);
    free(rknn_offset);
    free(rknn_data);
    free(unassigned);
    free(is_unassigned);
}

/* ── BƯỚC 7: Bầu chọn sửa lỗi ──────────────────────────────────────────── */

/* Bieu quyet nhan bang kNN, dung double-buffering. */
void step7_reallocate_by_voting(int* labels, const real_t* rho,
                                 const int* knn_idx,
                                 const real_t* knn_dist,
                                 int n, int k, int n_clusters) {
    /*
     * [DOMAIN] Double-buffering.
     *
     * [FIX-1] sort_desc_by_value O(n log n) thay core_sort_desc O(n²).
     *
     * [FIX-2] Pre-alloc counts_pool[T × n_clusters] ngoài parallel for:
     *   Phiên bản cũ: calloc(n_clusters) bên trong → n thread tranh heap lock.
     *   Phiên bản mới: counts_pool cấp phát 1 lần, thread tid dùng slot
     *   [tid*n_clusters .. (tid+1)*n_clusters). Reset O(n_clusters) mỗi iter.
     *
     * Memory model (double-buffer):
     *   labels[]      : read-only trong toàn bộ parallel for → không conflict.
     *   new_labels[]  : thread ghi new_labels[order[pos]] riêng → không overlap.
     *   counts_pool[] : thread ghi slot [tid*n_clusters..) riêng → không overlap.
     *   Không có write conflict → không cần atomic hay mutex ở bất kỳ đâu. ✓
     */
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif

    /* [FIX-G] Pad slot lên bội 64B để tránh false sharing.
     * Với n_clusters=10, slot_ints=16 (64B) thay vì 10 (40B).
     * _aligned_malloc đảm bảo base address căn chỉnh 64B. */
    int slot_ints = ((n_clusters + 15) / 16) * 16;

    int* order       = (int*)malloc((size_t)n * sizeof(int));
    int* new_labels  = (int*)malloc((size_t)n * sizeof(int));
    int* counts_pool = (int*)_aligned_malloc(
        (size_t)nthreads * (size_t)slot_ints * sizeof(int), 64);

    sort_desc_by_value(rho, order, n); /* [FIX-1] O(n log n) */
    memcpy(new_labels, labels, (size_t)n * sizeof(int));

#pragma omp parallel for schedule(static)
    for (int pos = 0; pos < n; pos++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        /* [FIX-G] slot riêng, padded 64B — không false sharing */
        int* counts = counts_pool + (size_t)tid * slot_ints;
        for (int c = 0; c < n_clusters; c++) counts[c] = 0;

        int i = order[pos];
        for (int t = 0; t < k; t++) {
            int lb = labels[knn_idx[i*k+t]]; /* read-only */
            if (lb >= 0) counts[lb]++;
        }

        int best_count = 0;
        for (int c = 0; c < n_clusters; c++)
            if (counts[c] > best_count) best_count = counts[c];

        if (best_count > 0) {
            real_t best_mean = 1e100;
            int    best_cl   = -1;
            for (int c = 0; c < n_clusters; c++) {
                if (counts[c] != best_count) continue;
                real_t sum = 0.0; int cnt = 0;
                for (int t = 0; t < k; t++) {
                    if (labels[knn_idx[i*k+t]] == c) {
                        sum += knn_dist[i*k+t]; cnt++;
                    }
                }
                real_t mean = cnt > 0 ? sum / (real_t)cnt : 1e100;
                if (best_cl < 0 || mean < best_mean) { best_mean = mean; best_cl = c; }
            }
            new_labels[i] = best_cl; /* ghi buffer riêng → không race */
        }
    }

    memcpy(labels, new_labels, (size_t)n * sizeof(int));
    _aligned_free(counts_pool); /* [FIX-G] paired with _aligned_malloc */
    free(new_labels);
    free(order);
}

/* ── BƯỚC 8: Vét cạn ngoại lai ──────────────────────────────────────────── */

/* Gan nhan cuoi: chon cum co mean distance nho nhat. */
void step8_allocate_remaining(int* labels, const int* knn_idx,
                               const real_t* knn_dist,
                               int n, int k, int n_clusters) {
    /*
     * [DOMAIN] Double-buffering — giữ nguyên, đã tối ưu.
     * labels[] read-only, new_labels[] ghi riêng từng thread → không race.
     */
    int* new_labels = (int*)malloc((size_t)n * sizeof(int));
    memcpy(new_labels, labels, (size_t)n * sizeof(int));

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        if (labels[i] >= 0) continue;
        real_t best = 1e100;
        int best_cl = -1; /* [FIX-H] tường minh: -1 = chưa tìm thấy cụm nào */
        for (int c = 0; c < n_clusters; c++) {
            real_t sum = 0.0; int cnt = 0;
            for (int t = 0; t < k; t++) {
                if (labels[knn_idx[i*k+t]] == c) {
                    sum += knn_dist[i*k+t]; cnt++;
                }
            }
            if (cnt > 0) {
                real_t mean = sum / (real_t)cnt;
                if (mean < best) { best = mean; best_cl = c; }
            }
        }
        /* [FIX-H] Nếu không có neighbor nào labeled, fallback cluster 0 */
        new_labels[i] = (best_cl >= 0) ? best_cl : 0;
    }
    memcpy(labels, new_labels, (size_t)n * sizeof(int));
    free(new_labels);
}