/*
 * dpc_aknn_core.h - Hàm tính toán lõi DPC-AKNN (không dùng D[n×n]).
 *
 * Thay đổi quan trọng so với phiên bản cũ:
 *   - Bỏ hoàn toàn ma trận D[n×n] (~39GB cho 70K mẫu).
 *   - Khoảng cách tính on-the-fly khi cần (đúng lý thuyết, O(1) bộ nhớ thêm).
 *   - Chỉ lưu knn_idx[n×k] và knn_dist[n×k] (~84MB cho 70K mẫu, k=15).
 *   - Bước 3b (delta) giờ song song được: [DOMAIN] thay vì [SERIAL].
 *
 * Quy ước song song hóa:
 *   [DOMAIN]  : Phân rã miền - chia N điểm cho các luồng.
 *   [SERIAL]  : Tuần tự - phụ thuộc dữ liệu.
 */
#ifndef DPC_AKNN_CORE_H
#define DPC_AKNN_CORE_H

#include "config.h"
#include <math.h>

/* Tính khoảng cách Euclid on-the-fly giữa điểm i và j trong X[n×d]. */
static inline real_t dist_euclid(const real_t* X, int i, int j, int d) {
    real_t sum = 0.0;
    for (int p = 0; p < d; p++) {
        real_t diff = X[i*d+p] - X[j*d+p];
        sum += diff * diff;
    }
    return sqrt(sum);
}

/* Bình phương khoảng cách Euclid (không sqrt) — dùng để so sánh trong heap.
 * Vì sqrt() là hàm đơn điệu tăng: a < b ↔ sqrt(a) < sqrt(b)
 * → thứ tự so sánh hoàn toàn giống với khoảng cách thật. */
static inline real_t dist_euclid_sq(const real_t* X, int i, int j, int d) {
    real_t sum = 0.0;
    for (int p = 0; p < d; p++) {
        real_t diff = X[i*d+p] - X[j*d+p];
        sum += diff * diff;
    }
    return sum;
}

/* Bình phương khoảng cách Euclid với Early Exit:
 * Nếu tổng bình phương từng chiều đã vượt ngưỡng `threshold`,
 * dừng ngay và trả về giá trị lớn hơn threshold (chắc chắn bị loại).
 * Kết quả KHÔNG XẤP XỈ: mọi điểm bị cắt sớm đều chắc chắn xa hơn
 * ngưỡng hiện tại — không bao giờ bỏ sót điểm gần. */
static inline real_t dist_euclid_sq_early(const real_t* X, int i, int j,
                                            int d, real_t threshold) {
    real_t sum = 0.0;
    for (int p = 0; p < d; p++) {
        real_t diff = X[i*d+p] - X[j*d+p];
        sum += diff * diff;
        if (sum > threshold) return sum; /* Early exit — chắc chắn bị loại */
    }
    return sum;
}

/* ── BƯỚC 1: kNN trực tiếp từ X (không qua D[n×n]) ───────────────────────
 * [DOMAIN] Mỗi luồng tính khoảng cách on-the-fly cho tập điểm của nó.
 * Bộ nhớ: O(n×k) output + O(n) working per thread (thay vì O(n²)). */
void step1_compute_knn(const real_t* X, int* knn_idx, real_t* knn_dist,
                        int n, int d, int k);

/* ── BƯỚC 2: d_c thích ứng ────────────────────────────────────────────────
 * [DOMAIN] Pha 1 song song; [SERIAL] Reduction tổng toàn cục. */
real_t step2_compute_dc(const real_t* knn_dist, int n, int k);

/* ── BƯỚC 3a: Mật độ cục bộ ρ ────────────────────────────────────────────
 * [DOMAIN] rho[i] chỉ phụ thuộc knn_dist[i*k..] — hoàn toàn độc lập.
 * Không cần D: knn_dist[i*k+t] là khoảng cách từ i đến láng giềng thứ t. */
void step3a_compute_rho(const real_t* knn_dist,
                         real_t* rho, real_t d_c, int n, int k);

/* ── BƯỚC 3b: Khoảng cách tương đối δ ────────────────────────────────────
 * [DOMAIN] Giờ song song được! Mỗi delta[i] = min_{j: rho[j]>rho[i]} dist(i,j).
 * Không cần D: tính dist(X[i],X[j]) on-the-fly. O(n²) time, O(1) extra RAM. */
void step3b_compute_delta(const real_t* X, const real_t* rho,
                           real_t* delta, int n, int d);

/* ── BƯỚC 4: Chọn tâm cụm ────────────────────────────────────────────────
 * [SERIAL] Sắp xếp γ=ρ×δ, lấy top n_clusters. */
void step4_select_centers(const real_t* rho, const real_t* delta,
                           real_t* gamma_out,
                           int n, int n_clusters, int* centers_out);

/* ── BƯỚC 5: Cụm nòng cốt ban đầu (BFS) ──────────────────────────────────
 * [SERIAL] BFS phụ thuộc nhãn vừa gán.
 * Không cần D: dùng knn_dist cho d_pq; tính dist(centroid,x_q) on-the-fly. */
void step5_build_initial_clusters(int* labels, const int* centers,
                                   const real_t* X,
                                   const int* knn_idx,
                                   const real_t* knn_dist,
                                   real_t d_c,
                                   int n, int d, int k, int n_clusters);

/* ── BƯỚC 6: Ma trận liên kết A ──────────────────────────────────────────
 * [DOMAIN] build_A song song; [SERIAL] gán nhãn tuần tự.
 * Không cần D: d_il đã có trong knn_dist[i*k+t]. */
void step6_association_loop(int* labels, const int* knn_idx,
                             const real_t* knn_dist, const real_t* rho,
                             int n, int k, int n_clusters);

/* ── BƯỚC 7: Bầu chọn sửa lỗi ───────────────────────────────────────────
 * [DOMAIN] Double-buffer an toàn race condition.
 * Không cần D: khoảng cách đến nb đã có trong knn_dist[i*k+t]. */
void step7_reallocate_by_voting(int* labels, const real_t* rho,
                                 const int* knn_idx,
                                 const real_t* knn_dist,
                                 int n, int k, int n_clusters);

/* ── BƯỚC 8: Vét cạn ngoại lai ───────────────────────────────────────────
 * [DOMAIN] Mỗi điểm -1 độc lập.
 * Không cần D: khoảng cách đến nb đã có trong knn_dist[i*k+t]. */
void step8_allocate_remaining(int* labels, const int* knn_idx,
                               const real_t* knn_dist,
                               int n, int k, int n_clusters);

/* ── Tiện ích dùng chung ──────────────────────────────────────────────────*/
void core_compute_centroid(const real_t* X, const int* labels,
                            int cluster_id, int n, int d, real_t* centroid);
void core_sort_desc(const real_t* values, int* order, int n);

#endif /* DPC_AKNN_CORE_H */
