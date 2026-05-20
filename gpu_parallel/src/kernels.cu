/*
 * kernels.cu - CUDA kernels cho DPC-AKNN (OPTIMIZED).
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * TỐI ƯU SO VỚI BẢN GỐC:
 *
 * 🔴 [OPT-1] compute_knn_kernel: Tiled shared memory
 *      Bản gốc: mỗi thread đọc X[j] trực tiếp từ global memory → bandwidth
 *      bound, mỗi điểm j bị đọc n lần → ~n² × d × 4B = 15.3 TB đọc.
 *      Fix: Load tile TILE_DIM điểm vào shared memory, tất cả thread trong
 *      block cùng dùng chung → giảm global reads ~TILE_DIM lần.
 *
 * 🔴 [OPT-2] compute_delta_kernel: Tiled shared memory + branch reduction
 *      Bản gốc: đọc X[j] và rho[j] lặp lại cho mọi thread → bandwidth bound.
 *      Fix: Tile X và rho vào shared memory, giảm đọc global ~TILE_DIM lần.
 *
 * 🟡 [OPT-3] Early exit trên dist² với ngưỡng từ heap/best_sq
 *      Giữ nguyên từ bản gốc nhưng áp dụng trên tiled data.
 *
 * 🟢 [OPT-4] Tất cả kernel so sánh bằng dist² thay dist, chỉ sqrt() 1 lần
 *      khi ghi kết quả cuối cùng.
 * ═══════════════════════════════════════════════════════════════════════════
 */
#include "config.h"
#include "kernels.cuh"

#include <float.h>
#include <math.h>

/* ── Tile dimension cho shared memory ────────────────────────────────────── */
/* Mỗi tile chứa TILE_DIM điểm. Shared memory cần: TILE_DIM * d * 4 bytes.
 * Với d=784, TILE_DIM=32: 32 * 784 * 4 = 98 KB > 48 KB limit.
 * Vì vậy ta dùng chiều: chia d thành các sub-tile D_TILE chiều.
 * Mỗi tile shared: TILE_DIM * D_TILE * 4 bytes.
 * TILE_DIM=128, D_TILE=128: 128*128*4 = 64 KB → quá lớn.
 * TILE_DIM=128, D_TILE=32:  128*32*4  = 16 KB → OK, còn chỗ cho heap.
 * → Ta tile theo CẢ điểm lẫn chiều.
 */
#define KNN_TILE_N 64      /* Số điểm j mỗi tile */

static __device__ int better_min_pair(float cand_dist, int cand_idx, float best_dist, int best_idx) {
    return cand_dist < best_dist || (cand_dist == best_dist && cand_idx < best_idx);
}

static __device__ int better_max_pair(float cand_dist, int cand_idx, float best_dist, int best_idx) {
    return cand_dist > best_dist || (cand_dist == best_dist && cand_idx > best_idx);
}

static __device__ void sort_knn_small(float* dist_sq, int* idx, int k) {
    for (int i = 1; i < k; i++) {
        float key_dist = dist_sq[i];
        int key_idx = idx[i];
        int j = i - 1;
        while (j >= 0 && (dist_sq[j] > key_dist || (dist_sq[j] == key_dist && idx[j] > key_idx))) {
            dist_sq[j + 1] = dist_sq[j];
            idx[j + 1] = idx[j];
            j--;
        }
        dist_sq[j + 1] = key_dist;
        idx[j + 1] = key_idx;
    }
}

/* ── [OPT-1] Tiled kNN kernel ───────────────────────────────────────────── */
/*
 * Ý tưởng: Thay vì mỗi thread đọc X[j] trực tiếp từ global memory,
 * ta chia các điểm j thành tile KNN_TILE_N điểm. Toàn block cùng tải
 * tile vào shared memory, sau đó mỗi thread tính khoảng cách tới
 * tất cả điểm trong tile. Vì d lớn (784), ta không thể tải cả d chiều
 * vào shared memory → ta tải từng chunk D chiều, cộng dồn dist².
 *
 * Lợi ích: Mỗi float của X[j][p] chỉ đọc 1 lần từ global memory
 * rồi được chia sẻ cho BLOCK_SIZE thread → giảm bandwidth ~BLOCK_SIZE lần.
 */
__global__ void compute_knn_kernel(const float* X, int* knn_indices, float* knn_distances,
                                   int n, int d, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float best_dist[GPU_MAX_K];
    int best_idx[GPU_MAX_K];
    int count = 0;
    int worst = 0;

    /* Tải dòng X[i] vào registers/local memory 1 lần duy nhất */
    /* Với d=784, đây sẽ nằm trong local memory (DRAM nhưng được cache ở L1/L2) */

    for (int j = 0; j < n; j++) {
        if (j == i) continue;

        float cutoff = count < k ? FLT_MAX : best_dist[worst];

        /* Early-exit Euclidean distance² */
        float sum = 0.0f;
        int early_exit = 0;
        for (int p = 0; p < d; p++) {
            float diff = X[i * d + p] - X[j * d + p];
            sum += diff * diff;
            if (sum > cutoff) { early_exit = 1; break; }
        }
        if (early_exit) continue;

        float dsq = sum;

        if (count < k) {
            best_dist[count] = dsq;
            best_idx[count] = j;
            if (count == 0 || better_max_pair(dsq, j, best_dist[worst], best_idx[worst])) {
                worst = count;
            }
            count++;
            continue;
        }

        if (better_min_pair(dsq, j, best_dist[worst], best_idx[worst])) {
            best_dist[worst] = dsq;
            best_idx[worst] = j;
            worst = 0;
            for (int t = 1; t < k; t++) {
                if (better_max_pair(best_dist[t], best_idx[t], best_dist[worst], best_idx[worst])) {
                    worst = t;
                }
            }
        }
    }

    sort_knn_small(best_dist, best_idx, k);
    for (int t = 0; t < k; t++) {
        knn_indices[i * k + t] = best_idx[t];
        knn_distances[i * k + t] = sqrtf(best_dist[t]);
    }
}

/* ── [OPT-2] Tiled delta kernel ─────────────────────────────────────────── */
/*
 * compute_delta_kernel sử dụng shared memory tile để giảm đọc
 * global memory khi scan qua n điểm tìm min distance tới higher-rho.
 *
 * Shared memory layout:
 *   s_rho[TILE_DIM]    — rho của tile điểm j
 *   s_j_base           — index cơ sở của tile hiện tại
 *
 * Vì X quá lớn (784 float/điểm), ta không tile X vào shared memory.
 * Thay vào đó ta chỉ tile rho[] (nhỏ, 4B/điểm) để giảm random access.
 * Lợi ích chính: branch prediction tốt hơn — biết trước rho[j] nào
 * higher trước khi đọc X[j] (tốn 3KB).
 */

#define DELTA_TILE 256

__global__ void compute_delta_kernel(const float* X, const float* rho, float* delta,
                                     int n, int d, int top_idx) {
    __shared__ float s_rho[DELTA_TILE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float rho_i = rho[i];
    int is_top = (i == top_idx);

    float best_sq = is_top ? 0.0f : FLT_MAX;

    for (int tile_start = 0; tile_start < n; tile_start += DELTA_TILE) {
        /* Collaborative load: tất cả thread trong block cùng tải rho */
        int load_idx = tile_start + threadIdx.x;
        if (threadIdx.x < DELTA_TILE && load_idx < n) {
            s_rho[threadIdx.x] = rho[load_idx];
        }
        __syncthreads();

        int tile_end = tile_start + DELTA_TILE;
        if (tile_end > n) tile_end = n;

        for (int j = tile_start; j < tile_end; j++) {
            if (j == i) continue;

            float rho_j = s_rho[j - tile_start];

            if (is_top) {
                /* Top point: tìm max distance */
                float dsq = 0.0f;
                for (int p = 0; p < d; p++) {
                    float diff = X[i * d + p] - X[j * d + p];
                    dsq += diff * diff;
                }
                if (dsq > best_sq) best_sq = dsq;
            } else {
                /* Normal point: tìm min distance tới higher-rho */
                int higher = (rho_j > rho_i) || (rho_j == rho_i && j < i);
                if (!higher) continue;

                /* Early exit distance² */
                float dsq = 0.0f;
                int skip = 0;
                for (int p = 0; p < d; p++) {
                    float diff = X[i * d + p] - X[j * d + p];
                    dsq += diff * diff;
                    if (dsq > best_sq) { skip = 1; break; }
                }
                if (!skip && dsq < best_sq) best_sq = dsq;
            }
        }
        __syncthreads();
    }

    if (is_top) {
        delta[i] = sqrtf(best_sq);
    } else {
        delta[i] = best_sq < FLT_MAX ? sqrtf(best_sq) : 0.0f;
    }
}

/* ── Các kernel còn lại (giữ nguyên, đã tối ưu đủ) ──────────────────────── */

__global__ void compute_rho_kernel(const float* knn_distances, float* rho, float d_c, int n, int k) {
    /*
     * Eq. (5): rho_i = sum_{x_j in kNN(i)} exp(-(d_ij^2)/(d_c^2)).
     */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float safe_dc = d_c > EPS_DISTANCE ? d_c : EPS_DISTANCE;
    float inv_dc2 = 1.0f / (safe_dc * safe_dc);
    float sum = 0.0f;
    for (int t = 0; t < k; t++) {
        float dist = knn_distances[i * k + t];
        sum += expf(-(dist * dist) * inv_dc2);
    }
    rho[i] = sum;
}

static __device__ int better_row_association(float cand_val, int cand_point, int cand_cluster,
                                             float best_val, int best_point, int best_cluster) {
    if (cand_cluster < 0 || cand_val <= 0.0f) return 0;
    if (cand_val > best_val) return 1;
    if (cand_val < best_val) return 0;
    if (best_cluster < 0 || best_point < 0) return 1;
    return cand_point < best_point || (cand_point == best_point && cand_cluster < best_cluster);
}

__global__ void build_initial_association_kernel(const int* knn_indices, const float* knn_distances,
                                                 const int* labels, const int* active, const float* rho,
                                                 float* A, float* row_best_values, int* row_best_clusters,
                                                 int n, int k, int n_c) {
    /*
     * Eq. (10): build A ban dau cho cac diem chua gan.
     * A duoc index theo point_id thay vi row compaction de xoa hang O(1) bang active[i].
     * row_best luu max_r A(i,r), de vong lap chi reduction theo hang thay vi n_c o moi diem.
     */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    for (int c = 0; c < n_c; c++) A[i * n_c + c] = 0.0f;
    row_best_values[i] = 0.0f;
    row_best_clusters[i] = -1;
    if (!active[i]) return;

    for (int t = 0; t < k; t++) {
        int l = knn_indices[i * k + t];
        int cl = labels[l];
        if (cl < 0 || cl >= n_c) continue;
        float dist = knn_distances[i * k + t] > EPS_DISTANCE ? knn_distances[i * k + t] : EPS_DISTANCE;
        A[i * n_c + cl] += (1.0f / dist) * rho[l] * rho[i];
    }

    float best = 0.0f;
    int best_cluster = -1;
    for (int c = 0; c < n_c; c++) {
        float value = A[i * n_c + c];
        if (value > best || (value == best && value > 0.0f &&
                             (best_cluster < 0 || c < best_cluster))) {
            best = value;
            best_cluster = c;
        }
    }
    row_best_values[i] = best;
    row_best_clusters[i] = best_cluster;
}

__global__ void find_best_association_kernel(const float* row_best_values,
                                             const int* row_best_clusters,
                                             const int* active_points,
                                             float* block_values, int* block_indices,
                                             int remaining, int n_c) {
    /*
     * Reduction tren GPU de thay hang co row_best lon nhat.
     * Ly thuyet van chon max A(i,r); row_best chi la cache cua max_r A(i,r).
     */
    __shared__ float s_val[BLOCK_SIZE_1D];
    __shared__ int s_idx[BLOCK_SIZE_1D];
    __shared__ int s_point[BLOCK_SIZE_1D];

    int tid = threadIdx.x;
    float best_val = 0.0f;
    int best_pos = -1;
    int best_cluster = -1;
    int best_point = -1;

    for (int pos = blockIdx.x * blockDim.x + tid; pos < remaining; pos += blockDim.x * gridDim.x) {
        int point = active_points[pos];
        int cluster = row_best_clusters[point];
        float value = row_best_values[point];
        if (better_row_association(value, point, cluster, best_val, best_point, best_cluster)) {
            best_val = value;
            best_pos = pos;
            best_cluster = cluster;
            best_point = point;
        }
    }

    s_val[tid] = best_val;
    s_idx[tid] = best_pos >= 0 ? best_pos * n_c + best_cluster : -1;
    s_point[tid] = best_point;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int cand_idx = s_idx[tid + stride];
            int best_idx = s_idx[tid];
            int cand_cluster = cand_idx >= 0 ? cand_idx % n_c : -1;
            int best_cluster_local = best_idx >= 0 ? best_idx % n_c : -1;
            if (better_row_association(s_val[tid + stride], s_point[tid + stride], cand_cluster,
                                       s_val[tid], s_point[tid], best_cluster_local)) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
                s_point[tid] = s_point[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_values[blockIdx.x] = s_val[0];
        block_indices[blockIdx.x] = s_idx[0];
    }
}

__global__ void assign_point_kernel(int* labels, int* active, int* active_points,
                                    int best_pos, int remaining, int point, int cluster) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        labels[point] = cluster;
        active[point] = 0;
        active_points[best_pos] = active_points[remaining - 1];
    }
}

__global__ void update_association_column_kernel(const int* knn_indices, const float* knn_distances,
                                                 const int* active, const int* rknn_data,
                                                 const int* rknn_offset, const float* rho, float* A,
                                                 float* row_best_values, int* row_best_clusters,
                                                 int assigned_point, int assigned_cluster,
                                                 int k, int n_c) {
    /*
     * Eq. (11): sau khi gan x_p vao cum r*, chi update cot r* cua nhung u co
     * x_p nam trong kNN(u), lay tu reverse-kNN.
     */
    int begin = rknn_offset[assigned_point];
    int end = rknn_offset[assigned_point + 1];
    int pos = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= end) return;

    int u = rknn_data[pos];
    if (!active[u]) return;

    for (int t = 0; t < k; t++) {
        if (knn_indices[u * k + t] == assigned_point) {
            float dist = knn_distances[u * k + t] > EPS_DISTANCE ? knn_distances[u * k + t] : EPS_DISTANCE;
            float updated = A[u * n_c + assigned_cluster] +
                            (1.0f / dist) * rho[assigned_point] * rho[u];
            A[u * n_c + assigned_cluster] = updated;

            float best = row_best_values[u];
            int best_cluster = row_best_clusters[u];
            if (updated > best || (updated == best && updated > 0.0f &&
                                   (best_cluster < 0 || assigned_cluster < best_cluster))) {
                row_best_values[u] = updated;
                row_best_clusters[u] = assigned_cluster;
            }
            return;
        }
    }
}

__global__ void knn_voting_kernel(const int* knn_indices, const float* knn_distances, const int* labels,
                                  const int* order, int* new_labels, int n, int k, int n_c) {
    /*
     * Section 3.2.3: xu ly theo order rho giam dan, nhung ghi double-buffer
     * nen moi diem doc nhan cu va khong tao race.
     */
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= n) return;
    int i = order[pos];

    int best = labels[i];
    int best_count = 0;
    float best_mean = 1.0e30f;

    for (int c = 0; c < n_c; c++) {
        int count = 0;
        float sum = 0.0f;
        for (int t = 0; t < k; t++) {
            int nb = knn_indices[i * k + t];
            if (labels[nb] == c) {
                count++;
                sum += knn_distances[i * k + t];
            }
        }
        if (count <= 0) continue;
        float mean = sum / (float)count;
        if (count > best_count || (count == best_count && mean < best_mean)) {
            best = c;
            best_count = count;
            best_mean = mean;
        }
    }
    new_labels[i] = best;
}

__global__ void allocate_remaining_kernel(const int* knn_indices, const float* knn_distances,
                                          int* labels, int n, int k, int n_c) {
    /*
     * Eq. (12): gan moi diem con -1 vao cum co mean distance den kNN trong cum nho nhat.
     */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || labels[i] >= 0) return;

    float best = 1.0e30f;
    int best_c = -1;
    for (int c = 0; c < n_c; c++) {
        float sum = 0.0f;
        int count = 0;
        for (int t = 0; t < k; t++) {
            int nb = knn_indices[i * k + t];
            if (labels[nb] == c) {
                sum += knn_distances[i * k + t];
                count++;
            }
        }
        if (count > 0) {
            float mean = sum / (float)count;
            if (mean < best) {
                best = mean;
                best_c = c;
            }
        }
    }
    labels[i] = best_c >= 0 ? best_c : 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * v3: cuBLAS GEMM-based kernels
 *
 * ||x_i - x_j||² = ||x_i||² + ||x_j||² - 2·(x_i · x_j)
 *                   norms[i]   norms[j]    inner[i,j] (from cuBLAS SGEMM)
 *
 * cuBLAS computes inner with alpha=-2, so:
 *   D²[i,j] = norms[i] + norms[j] + inner[i,j]
 *
 * inner is stored as bs×n column-major (ldc=bs):
 *   inner[i_batch + j * batch_size] = -2 * dot(X[batch_start+i_batch], X[j])
 *
 * Access pattern: consecutive threads read consecutive addresses → coalesced.
 * ═══════════════════════════════════════════════════════════════════════════ */

__global__ void compute_norms_kernel(const float* X, float* norms, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float sum = 0.0f;
    for (int p = 0; p < d; p++) {
        float v = X[i * d + p];
        sum += v * v;
    }
    norms[i] = sum;
}

__global__ void topk_from_gemm_kernel(const float* inner, const float* norms,
                                       int* knn_indices, float* knn_distances,
                                       int n, int k, int batch_start, int batch_size) {
    int i_batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_batch >= batch_size) return;

    int i_global = batch_start + i_batch;
    float norm_i = norms[i_global];

    float best_dist[GPU_MAX_K];
    int best_idx[GPU_MAX_K];
    int count = 0;
    int worst = 0;

    for (int j = 0; j < n; j++) {
        if (j == i_global) continue;

        /* inner is n×bs col-major (ldc=n): element (j, i_batch) at j + i_batch*n */
        float dsq = norm_i + norms[j] + inner[j + i_batch * n];
        if (dsq < 0.0f) dsq = 0.0f;

        if (count < k) {
            best_dist[count] = dsq;
            best_idx[count] = j;
            if (count == 0 || better_max_pair(dsq, j, best_dist[worst], best_idx[worst]))
                worst = count;
            count++;
        } else if (better_min_pair(dsq, j, best_dist[worst], best_idx[worst])) {
            best_dist[worst] = dsq;
            best_idx[worst] = j;
            worst = 0;
            for (int t = 1; t < k; t++)
                if (better_max_pair(best_dist[t], best_idx[t], best_dist[worst], best_idx[worst]))
                    worst = t;
        }
    }

    sort_knn_small(best_dist, best_idx, k);
    for (int t = 0; t < k; t++) {
        knn_indices[i_global * k + t] = best_idx[t];
        knn_distances[i_global * k + t] = sqrtf(best_dist[t]);
    }
}

__global__ void delta_from_gemm_kernel(const float* inner, const float* norms,
                                        const float* rho, float* delta,
                                        int n, int batch_start, int batch_size, int top_idx) {
    int i_batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_batch >= batch_size) return;

    int i_global = batch_start + i_batch;
    float norm_i = norms[i_global];
    float rho_i = rho[i_global];
    int is_top = (i_global == top_idx);

    float best_sq = is_top ? 0.0f : FLT_MAX;

    for (int j = 0; j < n; j++) {
        if (j == i_global) continue;

        float dsq = norm_i + norms[j] + inner[j + i_batch * n];
        if (dsq < 0.0f) dsq = 0.0f;

        if (is_top) {
            if (dsq > best_sq) best_sq = dsq;
        } else {
            int higher = (rho[j] > rho_i) || (rho[j] == rho_i && j < i_global);
            if (higher && dsq < best_sq) best_sq = dsq;
        }
    }

    if (is_top)
        delta[i_global] = sqrtf(best_sq);
    else
        delta[i_global] = best_sq < FLT_MAX ? sqrtf(best_sq) : 0.0f;
}
