/*
 * dpc_aknn_algo.h - Giao diện luồng thuật toán DPC-AKNN.
 *
 * Thay đổi: Bỏ trường D (ma trận khoảng cách n×n) khỏi struct.
 *           Thêm X_ref để step3b, step5 tính khoảng cách on-the-fly.
 */
#ifndef DPC_AKNN_ALGO_H
#define DPC_AKNN_ALGO_H

#include "config.h"

typedef struct {
    int      n_clusters;
    int      k;
    int      n;           /* Số điểm */
    int      d;           /* Số chiều */
    int*     labels;      /* Nhãn phân cụm [n] */
    int*     centers;     /* Chỉ số tâm cụm [n_clusters] */
    real_t*  rho;         /* Mật độ cục bộ [n] */
    real_t*  delta;       /* Khoảng cách tương đối [n] */
    real_t*  gamma;       /* γ = ρ×δ [n] */
    real_t   d_c;         /* Khoảng cách cắt thích ứng */
    /* D[n×n] ĐÃ BỎ — tiết kiệm O(n²) bộ nhớ (~39GB cho 70K mẫu) */
    int*     knn_idx;     /* Chỉ số k láng giềng [n×k] */
    real_t*  knn_dist;    /* Khoảng cách k láng giềng [n×k] */
} DPCAKNNModel;

void algo_init(DPCAKNNModel* m, int n_clusters, int k);
void algo_fit(DPCAKNNModel* m, const real_t* X, int n, int d);
void algo_save_labels(const DPCAKNNModel* m, const char* filepath);
void algo_free(DPCAKNNModel* m);

#endif /* DPC_AKNN_ALGO_H */
