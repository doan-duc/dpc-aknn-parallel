/*
 * Core DPC-AKNN computations using an O(n * k) neighbor representation.
 *
 * Distance-heavy stages evaluate squared distances directly from X and use
 * exact early termination when a candidate already exceeds the active bound.
 */
#include "dpc_aknn_core.h"

#include <math.h>
#include <malloc.h>  /* _aligned_malloc and _aligned_free on Windows. */
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Internal helpers. */

typedef struct { real_t dist; int idx; } Neighbor;

/* Sort neighbors by ascending distance, then by ascending index. */
static int cmp_neighbor_asc(const void* a, const void* b) {
    const Neighbor* x = (const Neighbor*)a;
    const Neighbor* y = (const Neighbor*)b;
    if (x->dist < y->dist) return -1;
    if (x->dist > y->dist) return  1;
    return x->idx - y->idx;
}

/*
 * Build a descending index order in O(n log n). Equal values are ordered by
 * ascending source index, and all comparison state remains local.
 */
typedef struct { real_t val; int idx; } ValIdx;

/* Sort value-index pairs by descending value, then ascending index. */
static int cmp_validx_desc(const void* a, const void* b) {
    const ValIdx* x = (const ValIdx*)a;
    const ValIdx* y = (const ValIdx*)b;
    if (y->val > x->val) return  1;
    if (y->val < x->val) return -1;
    return x->idx - y->idx; /* Prefer the smaller index for equal values. */
}

/* Populate order with indices sorted by descending values. */
static void sort_desc_by_value(const real_t* values, int* order, int n) {
    ValIdx* vi = (ValIdx*)malloc((size_t)n * sizeof(ValIdx));
    for (int i = 0; i < n; i++) { vi[i].val = values[i]; vi[i].idx = i; }
    qsort(vi, (size_t)n, sizeof(ValIdx), cmp_validx_desc);
    for (int i = 0; i < n; i++) order[i] = vi[i].idx;
    free(vi);
}

/* Sort a small index set by descending value using selection sort. */
void core_sort_desc(const real_t* values, int* order, int n) {
    /* This helper is intended for small arrays such as center lists. */
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (values[order[j]] > values[order[i]] ||
               (values[order[j]] == values[order[i]] && order[j] < order[i])) {
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }
}

/* Compute a cluster centroid from X and the current labels. */
void core_compute_centroid(const real_t* X, const int* labels,
                            int cluster_id, int n, int d, real_t* centroid) {
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

/*
 * A size-k max-heap retains the k smallest candidates seen so far. The root
 * is the current largest retained distance, so each better candidate replaces
 * the root in O(log k) time.
 */
/* Restore the max-heap property below index i. */
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

/* Insert one candidate into the max-heap. */
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

/* Replace the root and restore the max-heap property. */
static void heap_replace_root(Neighbor* h, int size, Neighbor val) {
    h[0] = val;
    heap_sift_down(h, size, 0);
}

/* Step 1: Compute k-nearest neighbors with a max-heap and exact pruning. */
void step1_compute_knn(const real_t* X, int* knn_idx, real_t* knn_dist,
                        int n, int d, int k) {
    /*
     * Each OpenMP iteration owns one output row and uses a thread-local heap.
     * The heap stores squared distances, preserving ordering without square
     * roots. Once full, its root is the pruning threshold for exact early
     * termination. Square roots are applied only to the final k distances.
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
                /* Fill the heap before applying a distance threshold. */
                dsq = dist_euclid_sq(X, i, j, d);
                Neighbor nb = { dsq, j };
                heap_push(heap, &heap_sz, nb);
            } else {
                /* Prune candidates that exceed the largest retained distance. */
                dsq = dist_euclid_sq_early(X, i, j, d, heap[0].dist);
                if (dsq < heap[0].dist ||
                    (dsq == heap[0].dist && j < heap[0].idx)) {
                    Neighbor nb = { dsq, j };
                    heap_replace_root(heap, heap_sz, nb);
                }
            }
        }
        /* Sort the retained neighbors by ascending distance. */
        qsort(heap, (size_t)heap_sz, sizeof(Neighbor), cmp_neighbor_asc);
        /* Convert the final squared distances to Euclidean distances. */
        for (int t = 0; t < k; t++) {
            knn_idx [i * k + t] = heap[t].idx;
            knn_dist[i * k + t] = sqrt(heap[t].dist);
        }
    }

    for (int t = 0; t < nthreads; t++) free(bufs[t]);
    free(bufs);
}

/* Step 2: Compute the adaptive cutoff distance from per-point kNN distances. */
real_t step2_compute_dc(const real_t* knn_dist, int n, int k) {
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
    /* Reduce the global mean and variance in parallel. */
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

/* Step 3a: Compute local density from kNN distances and d_c. */
void step3a_compute_rho(const real_t* knn_dist,
                         real_t* rho, real_t d_c, int n, int k) {
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

/* Step 3b: Find the nearest point with higher density for each point. */
void step3b_compute_delta(const real_t* X, const real_t* rho,
                           real_t* delta, int n, int d) {
    /*
     * Static scheduling distributes independent points predictably. Squared
     * distances avoid square roots during the search, and best_sq provides
     * an exact pruning threshold for each candidate.
     */
    int top = 0;
    for (int i = 1; i < n; i++)
        if (rho[i] > rho[top] || (rho[i] == rho[top] && i < top))
            top = i;

    /* Compute the top-density point's maximum distance in parallel. */
    real_t max_dist_sq = 0.0;
#pragma omp parallel for reduction(max:max_dist_sq) schedule(static)
    for (int j = 0; j < n; j++) {
        real_t dsq = dist_euclid_sq(X, top, j, d);
        if (dsq > max_dist_sq) max_dist_sq = dsq;
    }
    real_t max_dist = sqrt(max_dist_sq);
    delta[top] = max_dist;

    /* Compute the minimum distance to a point with higher density. */
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

/* Step 4: Compute gamma and select the highest-scoring centers. */
void step4_select_centers(const real_t* rho, const real_t* delta,
                           real_t* gamma_out,
                           int n, int n_clusters, int* centers_out) {
    for (int i = 0; i < n; i++) gamma_out[i] = rho[i] * delta[i];
    int* order = (int*)malloc((size_t)n * sizeof(int));
    sort_desc_by_value(gamma_out, order, n);
    for (int c = 0; c < n_clusters; c++) centers_out[c] = order[c];
    free(order);
}

/* Step 5: Build initial clusters with one-hop expansion and centroid checks. */
void step5_build_initial_clusters(int* labels, const int* centers,
                                   const real_t* X,
                                   const int* knn_idx,
                                   const real_t* knn_dist,
                                   real_t d_c,
                                   int n, int d, int k, int n_clusters) {
    /*
     * Expansion is serial because each accepted point changes the active
     * cluster. The queue contains only the center's k-nearest neighbors, so
     * newly accepted points do not trigger recursive expansion. csum keeps
     * centroid updates at O(d) per accepted point.
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
        /*
         * Enqueue every center neighbor, including points already claimed by
         * another cluster, so each seed can contribute its own neighbors.
         */
        for (int t = 0; t < k; t++) {
            int nb = knn_idx[center * k + t];
            if (labels[nb] == -1) {
                labels[nb] = c;
                for (int p = 0; p < d; p++) csum[p] += X[nb*d+p];
                cnt++;
            }
            queue[tail++] = nb; /* The seed queue contains only center neighbors. */
        }
        /* Initialize the centroid from the coordinate sum. */
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

                /* Update the centroid incrementally after accepting x_q. */
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

/*
 * Step 6: Assign remaining points using the association score from Eq. (11).
 *
 * A reverse-kNN index identifies rows affected by each new label. A is built
 * once, then only the affected association entries are updated. A is indexed
 * by [point_id * n_clusters], and is_unassigned supports constant-time removal
 * from the active set.
 */

/* Assign remaining labels and update affected reverse-kNN entries. */
void step6_association_loop(int* labels, const int* knn_idx,
                             const real_t* knn_dist, const real_t* rho,
                             int n, int k, int n_clusters) {
    /* Build rknn[j] = {i | j is in kNN(i)}. */
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

    /* Collect unassigned points. */
    int* unassigned = (int*)malloc((size_t)n * sizeof(int));
    int* is_unassigned = (int*)calloc((size_t)n, sizeof(int));
    int n_u = 0;
    for (int i = 0; i < n; i++)
        if (labels[i] < 0) { unassigned[n_u++] = i; is_unassigned[i] = 1; }
    if (n_u == 0) goto cleanup_step6;

    /* Build the initial association matrix in parallel. */
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

    /* Assign the strongest remaining association at each iteration. */
    for (int iter = 0; iter < n_u; iter++) {
        /* Find the best association with thread-local candidates. */
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

        /* Update Eq. (11) only for reverse neighbors of best_pt. */
        for (int r = rknn_offset[best_pt]; r < rknn_offset[best_pt + 1]; r++) {
            int u = rknn_data[r];
            if (!is_unassigned[u]) continue;
            /* Locate the stored distance from u to best_pt. */
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

/* Step 7: Refine labels with double-buffered kNN voting. */
void step7_reallocate_by_voting(int* labels, const real_t* rho,
                                 const int* knn_idx,
                                 const real_t* knn_dist,
                                 int n, int k, int n_clusters) {
    /*
     * labels remains read-only during the parallel region, while each thread
     * writes a distinct new_labels entry and uses a private counts_pool slot.
     * The slots are allocated once outside the loop to avoid allocator
     * contention.
     */
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif

    /* Pad each counter slot to a 64-byte boundary to avoid false sharing. */
    int slot_ints = ((n_clusters + 15) / 16) * 16;

    int* order       = (int*)malloc((size_t)n * sizeof(int));
    int* new_labels  = (int*)malloc((size_t)n * sizeof(int));
    int* counts_pool = (int*)_aligned_malloc(
        (size_t)nthreads * (size_t)slot_ints * sizeof(int), 64);

    sort_desc_by_value(rho, order, n);
    memcpy(new_labels, labels, (size_t)n * sizeof(int));

#pragma omp parallel for schedule(static)
    for (int pos = 0; pos < n; pos++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        /* Select this thread's cache-line-aligned counter slot. */
        int* counts = counts_pool + (size_t)tid * slot_ints;
        for (int c = 0; c < n_clusters; c++) counts[c] = 0;

        int i = order[pos];
        for (int t = 0; t < k; t++) {
            int lb = labels[knn_idx[i*k+t]]; /* labels is read-only in this loop. */
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
            new_labels[i] = best_cl; /* Each iteration owns one output element. */
        }
    }

    memcpy(labels, new_labels, (size_t)n * sizeof(int));
    _aligned_free(counts_pool);
    free(new_labels);
    free(order);
}

/* Step 8: Assign remaining points to the cluster with minimum mean distance. */
void step8_allocate_remaining(int* labels, const int* knn_idx,
                               const real_t* knn_dist,
                               int n, int k, int n_clusters) {
    /*
     * labels remains read-only while each iteration writes one independent
     * new_labels element.
     */
    int* new_labels = (int*)malloc((size_t)n * sizeof(int));
    memcpy(new_labels, labels, (size_t)n * sizeof(int));

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        if (labels[i] >= 0) continue;
        real_t best = 1e100;
        int best_cl = -1; /* No candidate cluster has been found yet. */
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
        /* Fall back to cluster 0 when no neighbor has a valid label. */
        new_labels[i] = (best_cl >= 0) ? best_cl : 0;
    }
    memcpy(labels, new_labels, (size_t)n * sizeof(int));
    free(new_labels);
}
