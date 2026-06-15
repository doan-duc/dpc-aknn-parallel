/* Public orchestration interface for the eight-step DPC-AKNN algorithm. */
#ifndef DPC_AKNN_ALGO_H
#define DPC_AKNN_ALGO_H

#include "config.h"

typedef struct {
    int      n_clusters;
    int      k;
    int      n;           /* Number of points. */
    int      d;           /* Number of dimensions. */
    int*     labels;      /* Cluster labels, shape [n]. */
    int*     centers;     /* Center indices, shape [n_clusters]. */
    real_t*  rho;         /* Local density values, shape [n]. */
    real_t*  delta;       /* Relative distance values, shape [n]. */
    real_t*  gamma;       /* Density-distance scores, shape [n]. */
    real_t   d_c;         /* Adaptive cutoff distance. */
    int*     knn_idx;     /* Neighbor indices, shape [n * k]. */
    real_t*  knn_dist;    /* Neighbor distances, shape [n * k]. */
} DPCAKNNModel;

void algo_init(DPCAKNNModel* m, int n_clusters, int k);
void algo_fit(DPCAKNNModel* m, const real_t* X, int n, int d);
void algo_save_labels(const DPCAKNNModel* m, const char* filepath);
void algo_free(DPCAKNNModel* m);

#endif /* DPC_AKNN_ALGO_H */
