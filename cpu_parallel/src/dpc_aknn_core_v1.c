/*
 * dpc_aknn_core.c - Tính toán lõi DPC-AKNN, không dùng D[n×n].
 *
 * Thay đổi chính:
 *   - Bỏ step1_pairwise_distance + step1_knn_from_distance.
 *   - Thêm step1_compute_knn: tính khoảng cách on-the-fly, chỉ lưu kNN.
 *   - step3b_compute_delta: giờ [DOMAIN] song song (không phụ thuộc thứ tự).
 *   - step3a, step5, step6, step7, step8: dùng knn_dist thay vì D.
 */
#include "dpc_aknn_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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
    for (int i = 0; i < n; i++) order[i] = i; // Khởi tạo mảng thứ tự
    for (int i = 0; i < n - 1; i++) // Vòng lặp bên ngoài cho Selection Sort
        for (int j = i + 1; j < n; j++) // Vòng lặp bên trong để tìm giá trị lớn hơn
            if (values[order[j]] > values[order[i]] ||
               (values[order[j]] == values[order[i]] && order[j] < order[i])) {
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp; // Đổi chỗ
            }
}

void core_compute_centroid(const real_t* X, const int* labels,
                            int cluster_id, int n, int d, real_t* centroid) {
    /* [SERIAL] Đọc toàn bộ labels[], không song song. */
    int count = 0;
    for (int p = 0; p < d; p++) centroid[p] = 0.0; // Khởi tạo vector 0 cho centroid
    for (int i = 0; i < n; i++) { // Duyệt qua toàn bộ n điểm
        if (labels[i] != cluster_id) continue; // Chỉ tính điểm thuộc cụm cluster_id
        for (int p = 0; p < d; p++) centroid[p] += X[i*d+p]; // Cộng dồn tọa độ
        count++;
    }
    if (count > 0)
        for (int p = 0; p < d; p++) centroid[p] /= (real_t)count; // Tính trung bình tọa độ
}

/* ── BƯỚC 1: kNN trực tiếp từ X ─────────────────────────────────────────── */

void step1_compute_knn(const real_t* X, int* knn_idx, real_t* knn_dist,
                        int n, int d, int k) {
    /*
     * [DOMAIN] Phân rã miền theo điểm i.
     * - Mỗi luồng tính on-the-fly khoảng cách từ i đến tất cả j.
     * - Không cấp phát D[n×n] toàn cục; chỉ dùng Neighbor[n] local mỗi thread.
     * - Bộ nhớ: O(n×k) output + O(n) working per thread.
     * - Thay thế hoàn toàn step1_pairwise_distance + step1_knn_from_distance.
     */
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) { // SONG SONG: Mỗi thread xử lý một tập điểm i
        Neighbor* row = (Neighbor*)malloc((size_t)n * sizeof(Neighbor));
        /* Tính khoảng cách on-the-fly từ i đến tất cả j */
        for (int j = 0; j < n; j++) { // Duyệt tất cả j để tính khoảng cách tới i
            row[j].dist = dist_euclid(X, i, j, d);
            row[j].idx  = j;
        }
        qsort(row, (size_t)n, sizeof(Neighbor), cmp_neighbor); // Sắp xếp tìm láng giềng
        /* row[0] là chính điểm i (dist=0), lấy từ row[1] */
        for (int t = 0; t < k; t++) { // Lưu k láng giềng gần nhất
            knn_idx [i * k + t] = row[t + 1].idx;
            knn_dist[i * k + t] = row[t + 1].dist;
        }
        free(row); // Giải phóng mảng tạm của thread
    }
}

/* ── BƯỚC 2: d_c ────────────────────────────────────────────────────────── */

real_t step2_compute_dc(const real_t* knn_dist, int n, int k) {
    /*
     * [DOMAIN] Pha 1: tính d_ci song song.
     * [SERIAL] Pha 2: reduction toàn cục (O(n), không đáng song song).
     * Không cần D: knn_dist[i*k..i*k+k-1] đủ để tính mean+std của điểm i.
     */
    real_t* d_ci = (real_t*)malloc((size_t)n * sizeof(real_t));

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) { // SONG SONG: Tính d_ci cho từng điểm
        real_t mean = 0.0;
        for (int t = 0; t < k; t++) mean += knn_dist[i*k+t]; // Tổng khoảng cách kNN
        mean /= (real_t)k;
        real_t var = 0.0;
        for (int t = 0; t < k; t++) { // Tính phương sai khoảng cách
            real_t diff = knn_dist[i*k+t] - mean;
            var += diff * diff;
        }
        d_ci[i] = mean + sqrt(var / (real_t)k); // d_ci = mean + std
    }

    real_t mean_c = 0.0;
    for (int i = 0; i < n; i++) mean_c += d_ci[i]; // TUẦN TỰ: Tổng d_ci toàn cục
    mean_c /= (real_t)n;

    real_t var_c = 0.0;
    for (int i = 0; i < n; i++) { // TUẦN TỰ: Phương sai d_ci toàn cục
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
     * [DOMAIN] Phân rã miền theo điểm i.
     * Không cần D: knn_dist[i*k+t] là khoảng cách từ i đến láng giềng thứ t.
     * Công thức: rho_i = sum_{t=0}^{k-1} exp(-knn_dist[i,t]^2 / d_c^2)
     * Đúng với lý thuyết Eq.(5).
     */
    real_t safe_dc = d_c > EPS_DISTANCE ? d_c : EPS_DISTANCE;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) { // SONG SONG: Tính rho cho từng điểm
        real_t sum = 0.0;
        for (int t = 0; t < k; t++) { // Tính tổng hàm kernel Gaussian trên k láng giềng
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
     * [DOMAIN] GIỜ SONG SONG được (khác phiên bản cũ dùng [SERIAL]).
     *
     * Lý do phiên bản cũ dùng [SERIAL]: nó dùng trick sắp xếp thứ tự rho để
     * duyệt dần, nhưng điều đó KHÔNG cần thiết theo lý thuyết.
     *
     * Lý thuyết (Eq.4): delta_i = min_{j: rho[j] > rho[i]} dist(x_i, x_j)
     * Với mỗi i, điều kiện "rho[j] > rho[i]" chỉ đọc rho[] (read-only),
     * và ghi vào delta[i] riêng của nó → hoàn toàn độc lập giữa các i.
     *
     * Không cần D: tính dist(X[i], X[j]) on-the-fly.
     * Tiebreaker khi rho[j] == rho[i]: j có index nhỏ hơn = "cao hơn".
     */

    /* Tìm điểm có rho lớn nhất (top) */
    int top = 0;
    for (int i = 1; i < n; i++) // TUẦN TỰ: Tìm cực đại rho
        if (rho[i] > rho[top] || (rho[i] == rho[top] && i < top))
            top = i;

    /* delta[top] = khoảng cách xa nhất đến bất kỳ điểm nào */
    real_t max_dist = 0.0;
    for (int j = 0; j < n; j++) { // TUẦN TỰ: Tính delta cho điểm đỉnh mật độ
        real_t dist = dist_euclid(X, top, j, d);
        if (dist > max_dist) max_dist = dist;
    }
    delta[top] = max_dist;

    /* [DOMAIN] Tính delta cho tất cả điểm còn lại song song */
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) { // SONG SONG: Tính delta cho n-1 điểm còn lại
        if (i == top) continue;
        real_t best = 1e100;
        for (int j = 0; j < n; j++) { // Tìm điểm j có rho lớn hơn i và gần i nhất
            /* j "cao hơn" i nếu rho[j] > rho[i], hoặc rho bằng nhau và j < i */
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
    /* [SERIAL] γ = ρ×δ, sắp xếp lấy top n_clusters. O(n) — không cần song song. */
    int* order = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) gamma_out[i] = rho[i] * delta[i]; // Tính gamma = rho * delta
    core_sort_desc(gamma_out, order, n); // Sắp xếp giảm dần tìm tâm cụm
    for (int c = 0; c < n_clusters; c++) centers_out[c] = order[c]; // Lấy top k làm tâm cụm
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
     * [SERIAL] BFS — phụ thuộc nhãn vừa gán, không thể song song.
     *
     * Không cần D:
     *   - Điều kiện 2 (d_pq ≤ mean kNN): dùng knn_dist[x_p*k+t] trực tiếp.
     *   - Điều kiện 3 (dist(centroid, x_q) ≤ d_c): tính on-the-fly từ X.
     */
    real_t* centroid = (real_t*)malloc((size_t)d * sizeof(real_t));
    int*    queue    = (int*)   malloc((size_t)n * sizeof(int));

    for (int i = 0; i < n; i++) labels[i] = -1; // Khởi tạo nhãn -1 (chưa gán)

    for (int c = 0; c < n_clusters; c++) { // Duyệt qua từng tâm cụm
        int center = centers[c];
        if (labels[center] == -1) labels[center] = c;

        int head = 0, tail = 0;
        for (int t = 0; t < k; t++) { // Thêm k láng giềng của tâm vào queue
            int nb = knn_idx[center * k + t];
            if (labels[nb] == -1) labels[nb] = c;
            queue[tail++] = nb;
        }
        core_compute_centroid(X, labels, c, n, d, centroid);

        while (head < tail) { // Loang cụm bằng BFS
            int x_p = queue[head++];

            /* Tính mean khoảng cách kNN của x_p (điều kiện 2) */
            real_t local_mean = 0.0;
            for (int t = 0; t < k; t++) local_mean += knn_dist[x_p * k + t];
            local_mean /= (real_t)k;

            for (int t = 0; t < k; t++) { // Kiểm tra láng giềng của x_p
                int x_q = knn_idx[x_p * k + t];

                /* Điều kiện 1: chưa có nhãn */
                if (labels[x_q] != -1) continue;

                /* Điều kiện 2: khoảng cách x_p→x_q ≤ mean kNN(x_p) */
                if (knn_dist[x_p * k + t] > local_mean) continue;

                /* Điều kiện 3: dist(centroid, x_q) ≤ d_c */
                real_t dist_to_centroid = 0.0;
                for (int p = 0; p < d; p++) { // Tính dist on-the-fly tới centroid
                    real_t diff = centroid[p] - X[x_q * d + p];
                    dist_to_centroid += diff * diff;
                }
                if (sqrt(dist_to_centroid) > d_c) continue;

                labels[x_q] = c; // Gán nhãn cụm
                if (tail < n) queue[tail++] = x_q; // Thêm vào queue để loang tiếp
                core_compute_centroid(X, labels, c, n, d, centroid); // Cập nhật centroid
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
     * [DOMAIN] Phân rã miền theo các điểm unassigned.
     * Không cần D: knn_dist[i*k+t] = khoảng cách từ i đến knn_idx[i*k+t].
     * Công thức Eq.(10): A(i,r) = sum_{l in kNN(i) ∩ C_r} (1/d_il)*rho_l*rho_i
     */
    memset(A, 0, (size_t)n_u * (size_t)n_clusters * sizeof(real_t));
#pragma omp parallel for schedule(static)
    for (int row = 0; row < n_u; row++) { // SONG SONG: Tính ma trận liên kết cho các điểm chưa gán
        int i = unassigned[row];
        for (int t = 0; t < k; t++) { // Duyệt kNN của điểm i
            int    l  = knn_idx [i * k + t];
            real_t dl = knn_dist[i * k + t];  /* = dist(i, l) */
            int    cl = labels[l];
            if (cl < 0) continue;
            real_t safe_d = dl > EPS_DISTANCE ? dl : EPS_DISTANCE;
            A[row * n_clusters + cl] += (1.0 / safe_d) * rho[l] * rho[i]; // Cộng dồn liên kết cụm cl
        }
    }
}

void step6_association_loop(int* labels, const int* knn_idx,
                             const real_t* knn_dist, const real_t* rho,
                             int n, int k, int n_clusters) {
    /* Vòng ngoài: [SERIAL]. Bên trong: build_association_matrix [DOMAIN]. */
    int*    unassigned = (int*)   malloc((size_t)n * sizeof(int));
    real_t* A          = (real_t*)malloc((size_t)n * (size_t)n_clusters * sizeof(real_t));

    while (1) {
        int n_u = 0;
        for (int i = 0; i < n; i++) // Lấy danh sách điểm chưa gán nhãn
            if (labels[i] < 0) unassigned[n_u++] = i;
        if (n_u == 0) break;

        build_association_matrix(labels, knn_idx, knn_dist, rho,
                                  unassigned, n_u, A, k, n_clusters);

        real_t best = 0.0;
        int best_row = -1, best_cl = -1;
        for (int row = 0; row < n_u; row++) // Tìm liên kết lớn nhất trong ma trận A
            for (int c = 0; c < n_clusters; c++)
                if (A[row * n_clusters + c] > best) {
                    best = A[row * n_clusters + c];
                    best_row = row; best_cl = c;
                }
        if (best_row < 0) break;
        labels[unassigned[best_row]] = best_cl; // Gán nhãn cho điểm có liên kết mạnh nhất
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
     * [DOMAIN] Double-buffering — đọc labels[], ghi new_labels[].
     * Không cần D: knn_dist[i*k+t] = dist(i, knn_idx[i*k+t]).
     */
    int* order      = (int*)malloc((size_t)n * sizeof(int));
    int* new_labels = (int*)malloc((size_t)n * sizeof(int));
    core_sort_desc(rho, order, n);
    memcpy(new_labels, labels, (size_t)n * sizeof(int));

#pragma omp parallel for schedule(static)
    for (int pos = 0; pos < n; pos++) { // SONG SONG: Bầu chọn nhãn cho từng điểm i
        int i = order[pos];
        int* counts = (int*)calloc((size_t)n_clusters, sizeof(int));
        for (int t = 0; t < k; t++) { // Đếm nhãn trong k láng giềng
            int lb = labels[knn_idx[i*k+t]];
            if (lb >= 0) counts[lb]++;
        }
        int best_count = 0;
        for (int c = 0; c < n_clusters; c++) // Tìm nhãn phổ biến nhất
            if (counts[c] > best_count) best_count = counts[c];

        if (best_count > 0) {
            real_t best_mean = 1e100;
            int best_cl = -1;
            for (int c = 0; c < n_clusters; c++) { // Phá vỡ tie-break bằng khoảng cách trung bình
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
            new_labels[i] = best_cl; // Ghi nhãn mới vào buffer tạm
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
     * [DOMAIN] Mỗi điểm -1 xử lý độc lập.
     * Không cần D: knn_dist[i*k+t] = dist(i, knn_idx[i*k+t]).
     * Công thức Eq.(12): r* = argmin_r mean_dist(i, kNN(i) ∩ C_r).
     *
     * FIX Race Condition: Dùng double-buffering — đọc từ labels[] (read-only),
     * ghi vào new_labels[] riêng biệt, rồi memcpy ngược lại sau khi xong.
     * Điều này đảm bảo kết quả không phụ thuộc thứ tự chạy của các thread,
     * giống cách xử lý ở step7_reallocate_by_voting.
     */
    int* new_labels = (int*)malloc((size_t)n * sizeof(int));
    memcpy(new_labels, labels, (size_t)n * sizeof(int)); // Sao chép trạng thái ban đầu

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) { // SONG SONG: Gán các điểm còn sót cho cụm gần nhất
        if (labels[i] >= 0) continue; // Chỉ xử lý điểm chưa có nhãn — đọc labels[] (read-only)
        real_t best = 1e100;
        int best_cl = 0;
        for (int c = 0; c < n_clusters; c++) { // Duyệt k láng giềng để tìm cụm c gần nhất
            real_t sum = 0.0; int cnt = 0;
            for (int t = 0; t < k; t++) {
                if (labels[knn_idx[i*k+t]] == c) { // Đọc labels[] không thay đổi trong vùng parallel
                    sum += knn_dist[i*k+t];
                    cnt++;
                }
            }
            if (cnt > 0) {
                real_t mean = sum / (real_t)cnt;
                if (mean < best) { best = mean; best_cl = c; }
            }
        }
        new_labels[i] = best_cl; // Ghi vào buffer riêng, tránh race condition
    }
    memcpy(labels, new_labels, (size_t)n * sizeof(int)); // Cập nhật lại kết quả cuối cùng
    free(new_labels);
}
