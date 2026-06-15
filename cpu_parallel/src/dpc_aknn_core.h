/*
 * Core DPC-AKNN computations.
 *
 * Distances are evaluated from the input matrix as needed. The persistent
 * neighbor representation contains only knn_idx and knn_dist, both with
 * shape [n * k].
 */
#ifndef DPC_AKNN_CORE_H
#define DPC_AKNN_CORE_H

#include "config.h"
#include <math.h>

/* Compute the Euclidean distance between rows i and j of X. */
static inline real_t dist_euclid(const real_t* X, int i, int j, int d) {
    real_t sum = 0.0;
    for (int p = 0; p < d; p++) {
        real_t diff = X[i*d+p] - X[j*d+p];
        sum += diff * diff;
    }
    return sqrt(sum);
}

/*
 * Compute squared Euclidean distance. Squared values preserve distance
 * ordering and avoid an unnecessary square root during comparisons.
 */
static inline real_t dist_euclid_sq(const real_t* X, int i, int j, int d) {
    real_t sum = 0.0;
    for (int p = 0; p < d; p++) {
        real_t diff = X[i*d+p] - X[j*d+p];
        sum += diff * diff;
    }
    return sum;
}

/*
 * Compute squared Euclidean distance with an exact early exit. Once the
 * partial sum exceeds threshold, the candidate cannot satisfy the current
 * bound and the remaining dimensions do not need to be evaluated.
 */
static inline real_t dist_euclid_sq_early(const real_t* X, int i, int j,
                                            int d, real_t threshold) {
    real_t sum = 0.0;
    for (int p = 0; p < d; p++) {
        real_t diff = X[i*d+p] - X[j*d+p];
        sum += diff * diff;
        if (sum > threshold) return sum; /* The candidate already exceeds the bound. */
    }
    return sum;
}

/*
 * Step 1: Compute k-nearest neighbors directly from X.
 * Rows are independent and distributed across OpenMP threads.
 */
void step1_compute_knn(const real_t* X, int* knn_idx, real_t* knn_dist,
                        int n, int d, int k);

/* Step 2: Compute the adaptive cutoff distance using parallel reductions. */
real_t step2_compute_dc(const real_t* knn_dist, int n, int k);

/*
 * Step 3a: Compute local density. Each rho[i] depends only on the k
 * distances stored for point i.
 */
void step3a_compute_rho(const real_t* knn_dist,
                         real_t* rho, real_t d_c, int n, int k);

/*
 * Step 3b: Compute relative distance. Each point independently searches for
 * the nearest point with higher density.
 */
void step3b_compute_delta(const real_t* X, const real_t* rho,
                           real_t* delta, int n, int d);

/* Step 4: Select the n_clusters largest density-distance scores. */
void step4_select_centers(const real_t* rho, const real_t* delta,
                           real_t* gamma_out,
                           int n, int n_clusters, int* centers_out);

/*
 * Step 5: Build initial core clusters with serial one-hop expansion.
 * Neighbor distances come from knn_dist; centroid distances use X.
 */
void step5_build_initial_clusters(int* labels, const int* centers,
                                   const real_t* X,
                                   const int* knn_idx,
                                   const real_t* knn_dist,
                                   real_t d_c,
                                   int n, int d, int k, int n_clusters);

/*
 * Step 6: Build association scores in parallel and assign labels
 * iteratively.
 */
void step6_association_loop(int* labels, const int* knn_idx,
                             const real_t* knn_dist, const real_t* rho,
                             int n, int k, int n_clusters);

/* Step 7: Refine labels with race-free, double-buffered neighbor voting. */
void step7_reallocate_by_voting(int* labels, const real_t* rho,
                                 const int* knn_idx,
                                 const real_t* knn_dist,
                                 int n, int k, int n_clusters);

/* Step 8: Assign remaining unlabeled points independently. */
void step8_allocate_remaining(int* labels, const int* knn_idx,
                               const real_t* knn_dist,
                               int n, int k, int n_clusters);

/* Shared helpers. */
void core_compute_centroid(const real_t* X, const int* labels,
                            int cluster_id, int n, int d, real_t* centroid);
void core_sort_desc(const real_t* values, int* order, int n);

#endif /* DPC_AKNN_CORE_H */
