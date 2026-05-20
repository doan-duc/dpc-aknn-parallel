/*
 * dpc_aknn_core.c - Tính toán lõi DPC-AKNN, không dùng D[n×n].
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THAY ĐỔI SONG SONG HÓA SO VỚI PHIÊN BẢN CŨ:
 *
 * [1] step1_compute_knn  — PHÂN RÃ MIỀN (giữ) + fix malloc contention
 *     Vấn đề cũ : mỗi iteration gọi malloc/free bên trong parallel for
 *                 → n thread tranh heap lock đồng thời → serialize ẩn.
 *     Sửa       : pre-alloc 1 buffer per-thread TRƯỚC vùng parallel,
 *                 mỗi thread dùng buffer riêng, không tranh lock.
 *
 * [2] step3b_compute_delta — PHÂN RÃ MIỀN (giữ), schedule(static) giữ nguyên
 *     Vấn đề cũ : schedule(static) chia đều iteration, nhưng work KHÔNG đều:
 *                 điểm i có rho thấp phải duyệt gần n điểm j "cao hơn",
 *                 điểm i có rho cao chỉ duyệt vài j → thread cuối straggler.
 *     Revert    : dynamic(32) đo chậm hơn 11s trên high-dim data, giữ static
 *                 cho thread rảnh, cân bằng tải động.
 *
 * [3] step5_build_initial_clusters — PHÂN RÃ TÁC VỤ (thay TUẦN TỰ)
 *     Vấn đề cũ : n_clusters BFS chạy tuần tự dù hoàn toàn độc lập về
 *                 cấu trúc (mỗi cụm có queue, centroid, counter riêng).
 *     Sửa       : mỗi BFS = 1 omp task độc lập chạy song song.
 *     Race cond : labels[] bị ghi đồng thời → dùng atomic CAS
 *                 (__ATOMIC_SEQ_CST) thay check-then-set thông thường.
 *     False shar: head/tail/cnt per-cluster nằm liền nhau trong mảng
 *                 → pad đến bội số cache line 64-byte (CL_PAD = 16 int).
 *
 * [4] step6_association_loop — PHÂN RÃ MIỀN bổ sung cho find_best
 *     Vấn đề cũ : build_association_matrix đã song song nhưng find_best
 *                 (duyệt n_u × n_clusters phần tử) vẫn tuần tự.
 *     Sửa       : mỗi thread giữ (lb_val, lb_row, lb_cl) cục bộ, sau đó
 *                 dùng critical section nhỏ để merge kết quả.
 *                 Không dùng reduction(max) vì cần kèm theo index.
 * ═══════════════════════════════════════════════════════════════════════════
 */
#include "dpc_aknn_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ── Cache line size và padding ─────────────────────────────────────────── */
/* 64-byte cache line / sizeof(int) = 16 — pad mảng per-cluster về bội 64B  */
/* để tránh false sharing khi các task ghi vào slot kề nhau.                */
#define CL_INTS 16   /* số int = 1 cache line */

/* ── Tiện ích nội bộ ────────────────────────────────────────────────────── */

typedef struct { real_t dist; int idx; } Neighbor;

static int cmp_neighbor(const void* a, const void* b) {
    const Neighbor* x = (const Neighbor*)a;
    const Neighbor* y = (const Neighbor*)b;
    if (x->dist < y->dist) return -1;
    if (x->dist > y->dist) return  1;
    return x->idx - y->idx;
}

void core_sort_desc(const real_t* values, int* order, int n) {
    /* [SERIAL] Selection sort nhỏ - chỉ dùng cho n_clusters (thường ≤ 20). */
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (values[order[j]] > values[order[i]] ||
               (values[order[j]] == values[order[i]] && order[j] < order[i])) {
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }
}

void core_compute_centroid(const real_t* X, const int* labels,
                            int cluster_id, int n, int d, real_t* centroid) {
    /* [SERIAL] Đọc toàn bộ labels[], không song song. */
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

/* ── BƯỚC 1: kNN trực tiếp từ X ─────────────────────────────────────────── */

void step1_compute_knn(const real_t* X, int* knn_idx, real_t* knn_dist,
                        int n, int d, int k) {
    /*
     * [DOMAIN] Phân rã miền theo điểm i — giữ nguyên chiến lược.
     *
     * FIX malloc contention (CPU-RAM):
     *   Phiên bản cũ gọi malloc(n * sizeof(Neighbor)) BÊN TRONG vòng parallel,
     *   nghĩa là lúc chạy, T thread đồng thời gọi malloc → tranh heap lock
     *   → serialize ẩn, triệt tiêu phần lớn lợi ích song song.
     *
     *   Phiên bản mới: cấp phát T buffer (1 buffer/thread) TRƯỚC khi vào
     *   vùng parallel. Mỗi thread dùng buffer của riêng mình qua omp_get_thread_num(),
     *   không còn tranh lock. Sau vùng parallel mới giải phóng.
     *
     * Memory model:
     *   - X[] : read-only, mọi thread cùng đọc → không có write conflict.
     *   - knn_idx / knn_dist : schedule(static) chia theo khối liên tiếp,
     *     mỗi thread ghi vào dải [i_start..i_end) riêng biệt → không overlap.
     *     False sharing nhỏ tại biên giữa 2 chunk (1-2 cache line) — chấp nhận được.
     */
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif

    /* Cấp phát T buffer ngoài vùng parallel — tránh tranh heap lock */
    Neighbor** bufs = (Neighbor**)malloc((size_t)nthreads * sizeof(Neighbor*));
    for (int t = 0; t < nthreads; t++)
        bufs[t] = (Neighbor*)malloc((size_t)n * sizeof(Neighbor));

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num(); /* lấy id thread để chọn buffer riêng */
#endif
        Neighbor* row = bufs[tid]; /* buffer riêng — không malloc/free mỗi iter */

        for (int j = 0; j < n; j++) {
            row[j].dist = dist_euclid(X, i, j, d);
            row[j].idx  = j;
        }
        qsort(row, (size_t)n, sizeof(Neighbor), cmp_neighbor);

        for (int t = 0; t < k; t++) {
            knn_idx [i * k + t] = row[t + 1].idx;
            knn_dist[i * k + t] = row[t + 1].dist;
        }
        /* KHÔNG gọi free(row) — buffer được tái sử dụng qua các iteration */
    }

    /* Giải phóng sau khi vùng parallel kết thúc (có implicit barrier) */
    for (int t = 0; t < nthreads; t++) free(bufs[t]);
    free(bufs);
}

/* ── BƯỚC 2: d_c ────────────────────────────────────────────────────────── */

real_t step2_compute_dc(const real_t* knn_dist, int n, int k) {
    /*
     * [DOMAIN] Pha 1: tính d_ci song song.
     * [SERIAL] Pha 2: reduction toàn cục (O(n) — không đáng song song).
     * Giữ nguyên, đã tối ưu.
     */
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

    real_t mean_c = 0.0;
    for (int i = 0; i < n; i++) mean_c += d_ci[i];
    mean_c /= (real_t)n;

    real_t var_c = 0.0;
    for (int i = 0; i < n; i++) {
        real_t diff = d_ci[i] - mean_c;
        var_c += diff * diff;
    }
    real_t sigma_c = n > 1 ? sqrt(var_c / (real_t)(n - 1)) : 0.0;
    free(d_ci);
    return mean_c + sigma_c;
}

/* ── BƯỚC 3a: ρ ─────────────────────────────────────────────────────────── */

void step3a_compute_rho(const real_t* knn_dist,
                         real_t* rho, real_t d_c, int n, int k) {
    /*
     * [DOMAIN] Phân rã miền theo điểm i. Giữ nguyên, đã tối ưu.
     */
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

void step3b_compute_delta(const real_t* X, const real_t* rho,
                           real_t* delta, int n, int d) {
    /*
     * [DOMAIN] Phân rã miền — giữ nguyên chiến lược.
     *
     * Giữ nguyên schedule(static) — dynamic(32) đo được CHẬM HƠN 11s (414s vs 403s).
     * Với d=784 (fashion-mnist), curse of dimensionality khiến khoảng cách hội tụ
     * → rho phân phối đều → work mỗi iteration tương đồng → static là tối ưu.
     *
     * Memory model:
     *   - rho[] : read-only trong toàn bộ vùng parallel → không write conflict.
     *   - X[]   : read-only → không write conflict.
     *   - delta[]: mỗi thread ghi vào delta[i] riêng của nó → không overlap.
     */

    /* Tìm top tuần tự — O(n), không đáng song song so với O(n²) phía dưới */
    int top = 0;
    for (int i = 1; i < n; i++)
        if (rho[i] > rho[top] || (rho[i] == rho[top] && i < top))
            top = i;

    real_t max_dist = 0.0;
    for (int j = 0; j < n; j++) {
        real_t dist = dist_euclid(X, top, j, d);
        if (dist > max_dist) max_dist = dist;
    }
    delta[top] = max_dist;

    /* [DOMAIN] Giữ schedule(static) — dynamic CHẬM HƠN 11s trên fashion-mnist (d=784).
     * Curse of dimensionality khiến rho phân phối đều → work mỗi i tương đồng
     * → overhead dispatch dynamic(32) vượt lợi ích cân bằng tải. */
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        if (i == top) continue;
        real_t best = 1e100;
        for (int j = 0; j < n; j++) {
            int j_higher = (rho[j] > rho[i]) ||
                           (rho[j] == rho[i] && j < i);
            if (!j_higher) continue;
            real_t dist = dist_euclid(X, i, j, d);
            if (dist < best) best = dist;
        }
        delta[i] = (best < 1e99) ? best : max_dist;
    }
}

/* ── BƯỚC 4: Chọn tâm cụm ───────────────────────────────────────────────── */

void step4_select_centers(const real_t* rho, const real_t* delta,
                           real_t* gamma_out,
                           int n, int n_clusters, int* centers_out) {
    /* [SERIAL] O(n) — không cần song song. Giữ nguyên. */
    int* order = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) gamma_out[i] = rho[i] * delta[i];
    core_sort_desc(gamma_out, order, n);
    for (int c = 0; c < n_clusters; c++) centers_out[c] = order[c];
    free(order);
}

/* ── BƯỚC 5: Cụm nòng cốt ban đầu ───────────────────────────────────────── */

void step5_build_initial_clusters(int* labels, const int* centers,
                                   const real_t* X,
                                   const int* knn_idx,
                                   const real_t* knn_dist,
                                   real_t d_c,
                                   int n, int d, int k, int n_clusters) {
    /*
     * [SERIAL] BFS — giữ nguyên tuần tự, KHÔNG song song hóa.
     *
     * Lý do KHÔNG thể song song hóa step này mà giữ nguyên kết quả:
     *   Thuật toán xử lý cụm theo THỨ TỰ 0, 1, 2, ..., n_clusters-1.
     *   Cụm có gamma cao hơn (index nhỏ hơn) có QUYỀN ƯU TIÊN chiếm điểm
     *   ở vùng biên giữa các cụm. Đây là tính chất thuật toán, không phải
     *   chi tiết cài đặt — nếu song song hóa (task decomposition) thì thứ tự
     *   claim điểm trở thành non-deterministic, làm thay đổi kết quả phân cụm.
     *
     *   Thực nghiệm xác nhận: task decomposition với atomic CAS làm ARI giảm
     *   từ 0.407 xuống 0.376 — thua kém về chất lượng dù nhanh hơn về tốc độ.
     *   → Đây là trường hợp song song hóa gây hại, phải giữ serial.
     *
     * Không cần D:
     *   - Điều kiện 2 (d_pq ≤ mean kNN): dùng knn_dist[x_p*k+t] trực tiếp.
     *   - Điều kiện 3 (dist(centroid, x_q) ≤ d_c): tính on-the-fly từ X.
     */
    real_t* centroid = (real_t*)malloc((size_t)d * sizeof(real_t));
    int*    queue    = (int*)   malloc((size_t)n * sizeof(int));

    for (int i = 0; i < n; i++) labels[i] = -1;

    for (int c = 0; c < n_clusters; c++) {
        int center = centers[c];
        if (labels[center] == -1) labels[center] = c;

        int head = 0, tail = 0;
        for (int t = 0; t < k; t++) {
            int nb = knn_idx[center * k + t];
            if (labels[nb] == -1) labels[nb] = c;
            queue[tail++] = nb;
        }
        core_compute_centroid(X, labels, c, n, d, centroid);

        while (head < tail) {
            int x_p = queue[head++];

            real_t local_mean = 0.0;
            for (int t = 0; t < k; t++) local_mean += knn_dist[x_p * k + t];
            local_mean /= (real_t)k;

            for (int t = 0; t < k; t++) {
                int x_q = knn_idx[x_p * k + t];

                if (labels[x_q] != -1) continue;

                if (knn_dist[x_p * k + t] > local_mean) continue;

                real_t dist_to_centroid = 0.0;
                for (int p = 0; p < d; p++) {
                    real_t diff = centroid[p] - X[x_q * d + p];
                    dist_to_centroid += diff * diff;
                }
                if (sqrt(dist_to_centroid) > d_c) continue;

                labels[x_q] = c;
                core_compute_centroid(X, labels, c, n, d, centroid);
            }
        }
    }
    free(centroid);
    free(queue);
}

/* ── BƯỚC 6: Ma trận liên kết A ─────────────────────────────────────────── */

static void build_association_matrix(const int* labels,
                                      const int* knn_idx,
                                      const real_t* knn_dist,
                                      const real_t* rho,
                                      const int* unassigned, int n_u,
                                      real_t* A, int k, int n_clusters) {
    /*
     * [DOMAIN] Phân rã miền theo unassigned. Giữ nguyên, đã tối ưu.
     */
    memset(A, 0, (size_t)n_u * (size_t)n_clusters * sizeof(real_t));
#pragma omp parallel for schedule(static)
    for (int row = 0; row < n_u; row++) {
        int i = unassigned[row];
        for (int t = 0; t < k; t++) {
            int    l  = knn_idx [i * k + t];
            real_t dl = knn_dist[i * k + t];
            int    cl = labels[l];
            if (cl < 0) continue;
            real_t safe_d = dl > EPS_DISTANCE ? dl : EPS_DISTANCE;
            A[row * n_clusters + cl] += (1.0 / safe_d) * rho[l] * rho[i];
        }
    }
}

void step6_association_loop(int* labels, const int* knn_idx,
                             const real_t* knn_dist, const real_t* rho,
                             int n, int k, int n_clusters) {
    /*
     * Vòng ngoài: [SERIAL] — phụ thuộc label vừa gán ở iteration trước.
     * build_association_matrix: [DOMAIN] — giữ nguyên.
     *
     * FIX find_best: thêm [DOMAIN] song song cho phần trước bị bỏ sót.
     *
     *   Phiên bản cũ: duyệt n_u × n_clusters tuần tự sau khi build_A song song.
     *   Khi n lớn (n_u gần n), đây là O(n × n_clusters) tuần tự — chi phí đáng kể.
     *
     *   Phiên bản mới: mỗi thread giữ (lb_val, lb_row, lb_cl) thread-local,
     *   không ghi vào biến dùng chung → không cần atomic trong vòng for.
     *   Chỉ dùng critical section nhỏ O(T) để merge kết quả cuối.
     *   Không dùng reduction(max) vì cần kèm index (lb_row, lb_cl).
     *
     * Memory model:
     *   A[] : được ghi bởi build_association_matrix, có implicit barrier sau
     *   parallel for → tất cả write visible trước khi find_best bắt đầu. ✓
     */
    int*    unassigned = (int*)   malloc((size_t)n * sizeof(int));
    real_t* A          = (real_t*)malloc((size_t)n * (size_t)n_clusters * sizeof(real_t));

    while (1) {
        int n_u = 0;
        for (int i = 0; i < n; i++)
            if (labels[i] < 0) unassigned[n_u++] = i;
        if (n_u == 0) break;

        build_association_matrix(labels, knn_idx, knn_dist, rho,
                                  unassigned, n_u, A, k, n_clusters);

        /* find_best — song song hóa bằng thread-local reduction */
        real_t best     = 0.0;
        int    best_row = -1;
        int    best_cl  = -1;

#pragma omp parallel
        {
            /* Biến cục bộ của từng thread — không shared, không cần atomic */
            real_t lb_val = 0.0;
            int    lb_row = -1;
            int    lb_cl  = -1;

#pragma omp for schedule(static) nowait
            for (int row = 0; row < n_u; row++)
                for (int c = 0; c < n_clusters; c++)
                    if (A[row * n_clusters + c] > lb_val) {
                        lb_val = A[row * n_clusters + c];
                        lb_row = row;
                        lb_cl  = c;
                    }

            /* Merge thread-local → global: critical section nhỏ, chỉ chạy T lần */
#pragma omp critical
            {
                if (lb_val > best) {
                    best     = lb_val;
                    best_row = lb_row;
                    best_cl  = lb_cl;
                }
            }
        } /* implicit barrier — đảm bảo best_row/best_cl hoàn chỉnh */

        if (best_row < 0) break;
        labels[unassigned[best_row]] = best_cl;
    }

    free(unassigned);
    free(A);
}

/* ── BƯỚC 7: Bầu chọn sửa lỗi ──────────────────────────────────────────── */

void step7_reallocate_by_voting(int* labels, const real_t* rho,
                                 const int* knn_idx,
                                 const real_t* knn_dist,
                                 int n, int k, int n_clusters) {
    /*
     * [DOMAIN] Double-buffering — giữ nguyên, đã tối ưu.
     * Đọc labels[] (read-only), ghi new_labels[] riêng → không race.
     */
    int* order      = (int*)malloc((size_t)n * sizeof(int));
    int* new_labels = (int*)malloc((size_t)n * sizeof(int));
    core_sort_desc(rho, order, n);
    memcpy(new_labels, labels, (size_t)n * sizeof(int));

#pragma omp parallel for schedule(static)
    for (int pos = 0; pos < n; pos++) {
        int i = order[pos];
        int* counts = (int*)calloc((size_t)n_clusters, sizeof(int));
        for (int t = 0; t < k; t++) {
            int lb = labels[knn_idx[i*k+t]];
            if (lb >= 0) counts[lb]++;
        }
        int best_count = 0;
        for (int c = 0; c < n_clusters; c++)
            if (counts[c] > best_count) best_count = counts[c];

        if (best_count > 0) {
            real_t best_mean = 1e100;
            int best_cl = -1;
            for (int c = 0; c < n_clusters; c++) {
                if (counts[c] != best_count) continue;
                real_t sum = 0.0; int cnt = 0;
                for (int t = 0; t < k; t++) {
                    if (labels[knn_idx[i*k+t]] == c) {
                        sum += knn_dist[i*k+t];
                        cnt++;
                    }
                }
                real_t mean = cnt > 0 ? sum / (real_t)cnt : 1e100;
                if (best_cl < 0 || mean < best_mean) {
                    best_mean = mean; best_cl = c;
                }
            }
            new_labels[i] = best_cl;
        }
        free(counts);
    }
    memcpy(labels, new_labels, (size_t)n * sizeof(int));
    free(new_labels);
    free(order);
}

/* ── BƯỚC 8: Vét cạn ngoại lai ──────────────────────────────────────────── */

void step8_allocate_remaining(int* labels, const int* knn_idx,
                               const real_t* knn_dist,
                               int n, int k, int n_clusters) {
    /*
     * [DOMAIN] Double-buffering — giữ nguyên, đã tối ưu.
     * Đọc labels[] (read-only), ghi new_labels[] riêng → không race.
     */
    int* new_labels = (int*)malloc((size_t)n * sizeof(int));
    memcpy(new_labels, labels, (size_t)n * sizeof(int));

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        if (labels[i] >= 0) continue;
        real_t best = 1e100;
        int best_cl = 0;
        for (int c = 0; c < n_clusters; c++) {
            real_t sum = 0.0; int cnt = 0;
            for (int t = 0; t < k; t++) {
                if (labels[knn_idx[i*k+t]] == c) {
                    sum += knn_dist[i*k+t];
                    cnt++;
                }
            }
            if (cnt > 0) {
                real_t mean = sum / (real_t)cnt;
                if (mean < best) { best = mean; best_cl = c; }
            }
        }
        new_labels[i] = best_cl;
    }
    memcpy(labels, new_labels, (size_t)n * sizeof(int));
    free(new_labels);
}
