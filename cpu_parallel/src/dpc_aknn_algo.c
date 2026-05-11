/*
 * dpc_aknn_algo.c - Luồng điều phối 8 bước thuật toán DPC-AKNN.
 *
 * Thay đổi so với phiên bản cũ:
 *   - Bỏ cấp phát D[n×n] (~39GB cho 70K mẫu).
 *   - Bước 1: gọi step1_compute_knn() thay vì 2 hàm cũ.
 *   - Bước 3b: truyền X thay vì D.
 *   - Bước 5,6,7,8: truyền knn_dist thay vì D.
 */
#include "dpc_aknn_algo.h"
#include "dpc_aknn_core.h"
#include "io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
static double algo_now(void) { return omp_get_wtime(); }
#else
#include <time.h>
static double algo_now(void) { return (double)clock() / CLOCKS_PER_SEC; }
#endif

#define LOG(fmt, ...) \
    do { if (LOG_LEVEL >= 1) io_log("[DPC-AKNN] " fmt "\n", ##__VA_ARGS__); } while(0)

void algo_init(DPCAKNNModel* m, int n_clusters, int k) {
    memset(m, 0, sizeof(*m));
    m->n_clusters = n_clusters;
    m->k          = k;
}

void algo_fit(DPCAKNNModel* m, const real_t* X, int n, int d) {
    if (n <= m->k || n <= m->n_clusters) {
        io_log("[DPC-AKNN] Loi: n=%d qua nho (k=%d, clusters=%d)\n",
                n, m->k, m->n_clusters);
        return;
    }
    m->n = n; m->d = d;

    /* Cấp phát — KHÔNG có D[n×n] nữa */
    m->labels   = (int*)   malloc((size_t)n * sizeof(int));
    m->centers  = (int*)   malloc((size_t)m->n_clusters * sizeof(int));
    m->rho      = (real_t*)malloc((size_t)n * sizeof(real_t));
    m->delta    = (real_t*)malloc((size_t)n * sizeof(real_t));
    m->gamma    = (real_t*)malloc((size_t)n * sizeof(real_t));
    m->knn_idx  = (int*)   malloc((size_t)n * (size_t)m->k * sizeof(int));
    m->knn_dist = (real_t*)malloc((size_t)n * (size_t)m->k * sizeof(real_t));

    double t0, t1;

    /* Bước 1: kNN trực tiếp từ X — không qua D[n×n] */
    LOG("Buoc 1/8: kNN truc tiep tu X (n=%d, d=%d, k=%d) [DOMAIN]...", n, d, m->k);
    t0 = algo_now();
    step1_compute_knn(X, m->knn_idx, m->knn_dist, n, d, m->k);
    t1 = algo_now();
    LOG("  -> Xong. (%.3f s)", t1 - t0);

    /* Bước 2 */
    LOG("Buoc 2/8: Tinh d_c thich ung [DOMAIN+SERIAL]...");
    t0 = algo_now();
    m->d_c = step2_compute_dc(m->knn_dist, n, m->k);
    t1 = algo_now();
    LOG("  -> d_c = %.6f (%.3f s)", m->d_c, t1 - t0);

    /* Bước 3a */
    LOG("Buoc 3/8a: Tinh rho [DOMAIN]...");
    t0 = algo_now();
    step3a_compute_rho(m->knn_dist, m->rho, m->d_c, n, m->k);
    t1 = algo_now();
    LOG("  -> Xong. (%.3f s)", t1 - t0);

    /* Bước 3b — giờ [DOMAIN] song song (không cần D) */
    LOG("Buoc 3/8b: Tinh delta [DOMAIN] (on-the-fly, khong can D[nxn])...");
    t0 = algo_now();
    step3b_compute_delta(X, m->rho, m->delta, n, d);
    t1 = algo_now();
    LOG("  -> Xong. (%.3f s)", t1 - t0);

    /* Bước 4 */
    LOG("Buoc 4/8: Chon %d tam cum [SERIAL]...", m->n_clusters);
    t0 = algo_now();
    step4_select_centers(m->rho, m->delta, m->gamma, n, m->n_clusters, m->centers);
    t1 = algo_now();
    LOG("  -> Xong. (%.3f s)", t1 - t0);

    /* Bước 5 */
    LOG("Buoc 5/8: Cum nong cot ban dau (BFS) [SERIAL]...");
    t0 = algo_now();
    step5_build_initial_clusters(m->labels, m->centers, X,
                                  m->knn_idx, m->knn_dist,
                                  m->d_c, n, d, m->k, m->n_clusters);
    t1 = algo_now();
    LOG("  -> Xong. (%.3f s)", t1 - t0);

    /* Bước 6 */
    LOG("Buoc 6/8: Ma tran lien ket A [DOMAIN+SERIAL]...");
    t0 = algo_now();
    step6_association_loop(m->labels, m->knn_idx, m->knn_dist, m->rho,
                            n, m->k, m->n_clusters);
    t1 = algo_now();
    LOG("  -> Xong. (%.3f s)", t1 - t0);

    /* Bước 7 */
    LOG("Buoc 7/8: Bau chon sua loi [DOMAIN]...");
    t0 = algo_now();
    step7_reallocate_by_voting(m->labels, m->rho, m->knn_idx, m->knn_dist,
                                n, m->k, m->n_clusters);
    t1 = algo_now();
    LOG("  -> Xong. (%.3f s)", t1 - t0);

    /* Bước 8 */
    LOG("Buoc 8/8: Vet can ngoai lai [DOMAIN]...");
    t0 = algo_now();
    step8_allocate_remaining(m->labels, m->knn_idx, m->knn_dist,
                              n, m->k, m->n_clusters);
    t1 = algo_now();
    LOG("  -> Xong. (%.3f s)", t1 - t0);
}

void algo_save_labels(const DPCAKNNModel* m, const char* filepath) {
    io_write_labels(filepath, m->labels, m->n);
}

void algo_free(DPCAKNNModel* m) {
    free(m->labels);  free(m->centers);
    free(m->rho);     free(m->delta);    free(m->gamma);
    free(m->knn_idx); free(m->knn_dist);
    memset(m, 0, sizeof(*m));
}
