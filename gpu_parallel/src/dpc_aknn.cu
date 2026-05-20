/*
 * dpc_aknn.cu - Host orchestration cho DPC-AKNN ban CUDA.
 *
 * Luong nay giu dung cac buoc trong THEORY.md, nhung tranh cac bottleneck cu:
 * - Khong tao/copy D[n x n] ve host de lay kNN.
 * - Delta tinh song song tren GPU theo Eq. (4).
 * - Association matrix A build mot lan, sau do update tang dan theo Eq. (11).
 */
#include "dpc_aknn.h"
#include "kernels.cuh"
#include "utils_gpu.h"
#include <cublas_v2.h>

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float value;
    int idx;
} ValIdx;

static int cmp_validx_desc(const void* a, const void* b) {
    const ValIdx* x = (const ValIdx*)a;
    const ValIdx* y = (const ValIdx*)b;
    if (y->value > x->value) return 1;
    if (y->value < x->value) return -1;
    return x->idx - y->idx;
}

static void sort_indices_desc(const float* values, int* order, int n) {
    /*
     * O(n log n), thay selection sort O(n^2) trong ban cu.
     * Dung cho gamma va rho-order cua buoc voting.
     */
    ValIdx* pairs = (ValIdx*)malloc((size_t)n * sizeof(ValIdx));
    for (int i = 0; i < n; i++) {
        pairs[i].value = values[i];
        pairs[i].idx = i;
    }
    qsort(pairs, (size_t)n, sizeof(ValIdx), cmp_validx_desc);
    for (int i = 0; i < n; i++) order[i] = pairs[i].idx;
    free(pairs);
}

static int top_rho_index_host(const float* rho, int n) {
    int top = 0;
    for (int i = 1; i < n; i++) {
        if (rho[i] > rho[top] || (rho[i] == rho[top] && i < top)) top = i;
    }
    return top;
}

float compute_dc_host(const float* knn_distances, int n, int k) {
    /*
     * Eq. (6), (8), (9), (7): d_ci = mean(kNN_i) + std(kNN_i),
     * d_c = mean(d_ci) + sample_std(d_ci).
     */
    float* d_ci = (float*)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        float mean = 0.0f;
        for (int t = 0; t < k; t++) mean += knn_distances[i * k + t];
        mean /= (float)k;
        float var = 0.0f;
        for (int t = 0; t < k; t++) {
            float diff = knn_distances[i * k + t] - mean;
            var += diff * diff;
        }
        d_ci[i] = mean + sqrtf(var / (float)k);
    }

    float mean_c = 0.0f;
    for (int i = 0; i < n; i++) mean_c += d_ci[i];
    mean_c /= (float)n;

    float var_c = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = d_ci[i] - mean_c;
        var_c += diff * diff;
    }
    float sigma_c = n > 1 ? sqrtf(var_c / (float)(n - 1)) : 0.0f;
    free(d_ci);
    return mean_c + sigma_c;
}

static void select_centers_host(DPCAKNN_GPU* model) {
    int* order = (int*)malloc((size_t)model->n * sizeof(int));
    for (int i = 0; i < model->n; i++) model->h_gamma[i] = model->h_rho[i] * model->h_delta[i];
    sort_indices_desc(model->h_gamma, order, model->n);
    for (int c = 0; c < model->n_clusters; c++) model->h_centers[c] = order[c];
    free(order);
}

static void build_initial_clusters_host(DPCAKNN_GPU* model, const float* X) {
    /*
     * Section 3.2.1: BFS khoi tao cum giu serial de bao toan thu tu claim nhan.
     * Bottleneck cu la tinh lai centroid bang cach scan toan bo n sau moi lan them diem.
     * Ban nay giu running sum nen update centroid O(d).
     */
    int n = model->n, d = model->d, k = model->k;
    int* queue = (int*)malloc((size_t)n * sizeof(int));
    float* csum = (float*)malloc((size_t)d * sizeof(float));
    float* cent = (float*)malloc((size_t)d * sizeof(float));

    for (int i = 0; i < n; i++) model->h_labels[i] = -1;

    for (int c = 0; c < model->n_clusters; c++) {
        int center = model->h_centers[c];
        int count = 0;
        for (int p = 0; p < d; p++) {
            csum[p] = 0.0f;
            cent[p] = 0.0f;
        }

        if (model->h_labels[center] == -1) {
            model->h_labels[center] = c;
            for (int p = 0; p < d; p++) csum[p] += X[center * d + p];
            count++;
        }

        int head = 0, tail = 0;
        for (int t = 0; t < k; t++) {
            int nb = model->h_knn_indices[center * k + t];
            if (model->h_labels[nb] == -1) {
                model->h_labels[nb] = c;
                for (int p = 0; p < d; p++) csum[p] += X[nb * d + p];
                count++;
            }
            queue[tail++] = nb;
        }
        if (count > 0) {
            for (int p = 0; p < d; p++) cent[p] = csum[p] / (float)count;
        }

        while (head < tail) {
            int pidx = queue[head++];
            float local_mean = 0.0f;
            for (int t = 0; t < k; t++) local_mean += model->h_knn_distances[pidx * k + t];
            local_mean /= (float)k;

            for (int t = 0; t < k; t++) {
                int q = model->h_knn_indices[pidx * k + t];
                if (model->h_labels[q] != -1) continue;
                if (model->h_knn_distances[pidx * k + t] > local_mean) continue;

                float dist_sq = 0.0f;
                for (int dim = 0; dim < d; dim++) {
                    float diff = cent[dim] - X[q * d + dim];
                    dist_sq += diff * diff;
                }
                if (dist_sq > model->d_c * model->d_c) continue;

                model->h_labels[q] = c;
                if (tail < n) queue[tail++] = q;

                for (int dim = 0; dim < d; dim++) csum[dim] += X[q * d + dim];
                count++;
                for (int dim = 0; dim < d; dim++) cent[dim] = csum[dim] / (float)count;
            }
        }
    }

    free(queue);
    free(csum);
    free(cent);
}

static int collect_active_unassigned(const int* labels, int* active, int* active_points, int n) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        active[i] = labels[i] < 0 ? 1 : 0;
        if (active[i]) active_points[count++] = i;
    }
    return count;
}

static void build_reverse_knn_host(const int* knn_indices, int n, int k,
                                   int** offsets_out, int** data_out, int* total_out) {
    /*
     * rknn[j] = { i | j thuoc kNN(i) }, dung de update Eq. (11) chi tren cot r*.
     */
    int* counts = (int*)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < n; i++) {
        for (int t = 0; t < k; t++) counts[knn_indices[i * k + t]]++;
    }

    int* offsets = (int*)malloc(((size_t)n + 1) * sizeof(int));
    offsets[0] = 0;
    for (int i = 0; i < n; i++) offsets[i + 1] = offsets[i] + counts[i];

    int total = offsets[n];
    int* data = (int*)malloc((size_t)total * sizeof(int));
    memset(counts, 0, (size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) {
        for (int t = 0; t < k; t++) {
            int j = knn_indices[i * k + t];
            data[offsets[j] + counts[j]++] = i;
        }
    }

    free(counts);
    *offsets_out = offsets;
    *data_out = data;
    *total_out = total;
}

struct HeapNode {
    float val;
    int point;
};

struct MaxHeap {
    HeapNode* data;
    int* pos; // pos[point] = index in data array
    int size;
};

static void heap_swap(MaxHeap* h, int i, int j) {
    HeapNode tmp = h->data[i];
    h->data[i] = h->data[j];
    h->data[j] = tmp;
    h->pos[h->data[i].point] = i;
    h->pos[h->data[j].point] = j;
}

static void heap_sift_up(MaxHeap* h, int i) {
    while (i > 0) {
        int parent = (i - 1) / 2;
        int swap = 0;
        if (h->data[i].val > h->data[parent].val) {
            swap = 1;
        } else if (h->data[i].val == h->data[parent].val) {
            if (h->data[i].point < h->data[parent].point) {
                swap = 1;
            }
        }
        if (!swap) break;
        heap_swap(h, i, parent);
        i = parent;
    }
}

static void heap_sift_down(MaxHeap* h, int i) {
    while (2 * i + 1 < h->size) {
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        int largest = i;

        if (h->data[left].val > h->data[largest].val ||
            (h->data[left].val == h->data[largest].val && h->data[left].point < h->data[largest].point)) {
            largest = left;
        }
        if (right < h->size && (h->data[right].val > h->data[largest].val ||
            (h->data[right].val == h->data[largest].val && h->data[right].point < h->data[largest].point))) {
            largest = right;
        }

        if (largest == i) break;
        heap_swap(h, i, largest);
        i = largest;
    }
}

static void heap_push(MaxHeap* h, int point, float val) {
    int i = h->size++;
    h->data[i].point = point;
    h->data[i].val = val;
    h->pos[point] = i;
    heap_sift_up(h, i);
}

static void heap_update(MaxHeap* h, int point, float new_val) {
    int i = h->pos[point];
    if (i < 0 || i >= h->size) return;
    float old_val = h->data[i].val;
    h->data[i].val = new_val;
    if (new_val > old_val) {
        heap_sift_up(h, i);
    } else {
        heap_sift_down(h, i);
    }
}

static HeapNode heap_pop(MaxHeap* h) {
    HeapNode root = h->data[0];
    h->pos[root.point] = -1;
    if (h->size > 1) {
        h->data[0] = h->data[h->size - 1];
        h->pos[h->data[0].point] = 0;
        h->size--;
        heap_sift_down(h, 0);
    } else {
        h->size = 0;
    }
    return root;
}

static void association_loop_cpu(DPCAKNN_GPU* model) {
    int n = model->n, k = model->k, n_c = model->n_clusters;
    int* h_active = (int*)malloc((size_t)n * sizeof(int));
    int* h_active_points = (int*)malloc((size_t)n * sizeof(int));
    int remaining = collect_active_unassigned(model->h_labels, h_active, h_active_points, n);
    if (remaining == 0) {
        free(h_active_points);
        free(h_active);
        return;
    }

    int* h_rknn_offset = NULL;
    int* h_rknn_data = NULL;
    int total_rknn = 0;
    build_reverse_knn_host(model->h_knn_indices, n, k, &h_rknn_offset, &h_rknn_data, &total_rknn);

    float* h_A = (float*)calloc((size_t)n * (size_t)n_c, sizeof(float));
    float* h_row_best_values = (float*)calloc((size_t)n, sizeof(float));
    int* h_row_best_clusters = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) h_row_best_clusters[i] = -1;

    // MaxHeap allocation
    MaxHeap heap;
    heap.data = (HeapNode*)malloc((size_t)n * sizeof(HeapNode));
    heap.pos = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) heap.pos[i] = -1;
    heap.size = 0;

    // Initialize A and row bests for unassigned points
    for (int idx = 0; idx < remaining; idx++) {
        int i = h_active_points[idx];
        
        for (int t = 0; t < k; t++) {
            int l = model->h_knn_indices[i * k + t];
            int cl = model->h_labels[l];
            if (cl < 0) continue;
            float dist = model->h_knn_distances[i * k + t] > EPS_DISTANCE ? model->h_knn_distances[i * k + t] : EPS_DISTANCE;
            h_A[i * n_c + cl] += (1.0f / dist) * model->h_rho[l] * model->h_rho[i];
        }

        float best = 0.0f;
        int best_c = -1;
        for (int c = 0; c < n_c; c++) {
            float val = h_A[i * n_c + c];
            if (val > best || (val == best && val > 0.0f && (best_c < 0 || c < best_c))) {
                best = val;
                best_c = c;
            }
        }
        h_row_best_values[i] = best;
        h_row_best_clusters[i] = best_c;

        if (best > 0.0f && best_c >= 0) {
            heap_push(&heap, i, best);
        }
    }

    while (heap.size > 0) {
        HeapNode root = heap_pop(&heap);
        int best_point = root.point;
        int best_cluster = h_row_best_clusters[best_point];

        model->h_labels[best_point] = best_cluster;
        h_active[best_point] = 0;

        int begin = h_rknn_offset[best_point];
        int end = h_rknn_offset[best_point + 1];
        for (int pos = begin; pos < end; pos++) {
            int u = h_rknn_data[pos];
            if (!h_active[u]) continue;

            for (int t = 0; t < k; t++) {
                if (model->h_knn_indices[u * k + t] == best_point) {
                    float dist = model->h_knn_distances[u * k + t] > EPS_DISTANCE ? model->h_knn_distances[u * k + t] : EPS_DISTANCE;
                    float updated = h_A[u * n_c + best_cluster] + (1.0f / dist) * model->h_rho[best_point] * model->h_rho[u];
                    h_A[u * n_c + best_cluster] = updated;

                    float best_u = h_row_best_values[u];
                    int best_c_u = h_row_best_clusters[u];
                    if (updated > best_u || (updated == best_u && updated > 0.0f && (best_c_u < 0 || best_cluster < best_c_u))) {
                        h_row_best_values[u] = updated;
                        h_row_best_clusters[u] = best_cluster;
                        
                        if (heap.pos[u] == -1) {
                            heap_push(&heap, u, updated);
                        } else {
                            heap_update(&heap, u, updated);
                        }
                    }
                    break;
                }
            }
        }
    }

    free(h_active);
    free(h_active_points);
    free(h_rknn_offset);
    free(h_rknn_data);
    free(h_A);
    free(h_row_best_values);
    free(h_row_best_clusters);
    free(heap.data);
    free(heap.pos);
}

static void voting_gpu(DPCAKNN_GPU* model) {
    /*
     * Section 3.2.3: sap theo rho giam dan roi bo phieu kNN.
     * Kernel ghi vao new_labels de tranh race va khong phu thuoc thu tu block.
     */
    int n = model->n;
    int* h_order = (int*)malloc((size_t)n * sizeof(int));
    sort_indices_desc(model->h_rho, h_order, n);

    int* d_order = (int*)gpu_malloc_check((size_t)n * sizeof(int), "d_order");
    int* d_new = (int*)gpu_malloc_check((size_t)n * sizeof(int), "d_new_labels");
    CUDA_CHECK(cudaMemcpy(model->d_labels, model->h_labels, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_order, h_order, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));

    int block = BLOCK_SIZE_1D;
    int grid = (n + block - 1) / block;
    knn_voting_kernel<<<grid, block>>>(model->d_knn_indices, model->d_knn_distances, model->d_labels,
                                      d_order, d_new, n, model->k, model->n_clusters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(model->h_labels, d_new, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_order);
    cudaFree(d_new);
    free(h_order);
}

static void allocate_remaining_gpu(DPCAKNN_GPU* model) {
    int n = model->n;
    CUDA_CHECK(cudaMemcpy(model->d_labels, model->h_labels, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));

    int block = BLOCK_SIZE_1D;
    int grid = (n + block - 1) / block;
    allocate_remaining_kernel<<<grid, block>>>(model->d_knn_indices, model->d_knn_distances,
                                               model->d_labels, n, model->k, model->n_clusters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(model->h_labels, model->d_labels, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));
}

void dpcaknn_gpu_init(DPCAKNN_GPU* model, int n_clusters, int k) {
    memset(model, 0, sizeof(*model));
    model->n_clusters = n_clusters;
    model->k = k;
}

void dpcaknn_gpu_fit(DPCAKNN_GPU* model, const float* h_X, int n, int d) {
    if (model->k <= 0 || model->n_clusters <= 0) {
        fprintf(stderr, "Tham so khong hop le: k=%d, clusters=%d\n", model->k, model->n_clusters);
        exit(1);
    }
    if (n <= model->k || n <= model->n_clusters) {
        fprintf(stderr, "Du lieu qua nho: n=%d, k=%d, clusters=%d\n", n, model->k, model->n_clusters);
        exit(1);
    }
    if (model->k > GPU_MAX_K) {
        fprintf(stderr, "k=%d vuot GPU_MAX_K=%d; tang GPU_MAX_K trong config.h neu can\n",
                model->k, GPU_MAX_K);
        exit(1);
    }

    model->n = n;
    model->d = d;
    model->h_labels = (int*)malloc((size_t)n * sizeof(int));
    model->h_centers = (int*)malloc((size_t)model->n_clusters * sizeof(int));
    model->h_rho = (float*)malloc((size_t)n * sizeof(float));
    model->h_delta = (float*)malloc((size_t)n * sizeof(float));
    model->h_gamma = (float*)malloc((size_t)n * sizeof(float));
    model->h_knn_indices = (int*)malloc((size_t)n * (size_t)model->k * sizeof(int));
    model->h_knn_distances = (float*)malloc((size_t)n * (size_t)model->k * sizeof(float));

    // Initialize CUDA runtime and validate device index early.
    int device_count = 0;
    cudaError_t dev_err = cudaGetDeviceCount(&device_count);
    if (dev_err != cudaSuccess) {
        const char* err_name = cudaGetErrorName(dev_err);
        const char* err_str = cudaGetErrorString(dev_err);
        if (!err_name) err_name = "unknown CUDA error";
        if (!err_str) err_str = "unknown CUDA error";
        fprintf(stderr, "CUDA init failed in cudaGetDeviceCount: %s (%s, code=%d)\n",
                err_str, err_name, (int)dev_err);
        exit(1);
    }
    if (device_count <= 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        exit(1);
    }
    if (GPU_DEVICE_ID < 0 || GPU_DEVICE_ID >= device_count) {
        fprintf(stderr, "GPU_DEVICE_ID=%d out of range (0..%d).\n", GPU_DEVICE_ID, device_count - 1);
        exit(1);
    }
    CUDA_CHECK(cudaSetDevice(GPU_DEVICE_ID));
    model->d_X = (float*)gpu_malloc_check((size_t)n * (size_t)d * sizeof(float), "d_X");
    model->d_rho = (float*)gpu_malloc_check((size_t)n * sizeof(float), "d_rho");
    model->d_delta = (float*)gpu_malloc_check((size_t)n * sizeof(float), "d_delta");
    model->d_labels = (int*)gpu_malloc_check((size_t)n * sizeof(int), "d_labels");
    model->d_knn_indices = (int*)gpu_malloc_check((size_t)n * (size_t)model->k * sizeof(int), "d_knn_indices");
    model->d_knn_distances = (float*)gpu_malloc_check((size_t)n * (size_t)model->k * sizeof(float), "d_knn_distances");

    CUDA_CHECK(cudaMemcpy(model->d_X, h_X, (size_t)n * (size_t)d * sizeof(float), cudaMemcpyHostToDevice));

    int block = BLOCK_SIZE_1D;
    int grid = (n + block - 1) / block;

    double start_step, end_step;

    /* ── cuBLAS setup ──────────────────────────────────────────────────── */
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    /* Precompute norms: norms[i] = ||X[i]||² */
    float* d_norms = (float*)gpu_malloc_check((size_t)n * sizeof(float), "d_norms");
    compute_norms_kernel<<<grid, block>>>(model->d_X, d_norms, n, d);
    CUDA_CHECK(cudaGetLastError());

    /* Batch size for GEMM: inner buffer = bs × n × 4 bytes */
    int gemm_bs = GPU_BATCH_SIZE;  /* from config.h, default 5000 */
    /* Ensure batch fits in ~1.5 GB: bs < 1.5GB / (n*4) */
    while ((size_t)gemm_bs * (size_t)n * sizeof(float) > (size_t)1500 * 1024 * 1024 && gemm_bs > 1000)
        gemm_bs /= 2;
    float* d_inner = (float*)gpu_malloc_check((size_t)gemm_bs * (size_t)n * sizeof(float), "d_inner");
    log_printf("[DPC-AKNN] cuBLAS GEMM batch size: %d (inner buffer: %.1f MB)\n",
               gemm_bs, (double)gemm_bs * n * sizeof(float) / (1024.0 * 1024.0));

    /*
     * Buoc 1: kNN via cuBLAS GEMM.
     * ||x_i - x_j||² = norms[i] + norms[j] - 2·dot(x_i, x_j)
     * cuBLAS computes: Inner = -2 · X_batch · X^T
     */
    log_printf("[DPC-AKNN] Buoc 1/8: kNN via cuBLAS GEMM (n=%d, d=%d, k=%d) [GPU]...\n", n, d, model->k);
    start_step = get_time_sec();
    {
        float alpha = -2.0f, beta_val = 0.0f;
        for (int b = 0; b < n; b += gemm_bs) {
            int bs = (b + gemm_bs <= n) ? gemm_bs : (n - b);
            /* cublasSgemm: C[n×bs col-major] = alpha * op(A)[n×d] * op(B)[d×bs]
             * A = d_X (d×n col-major = n×d row-major), transa=T → n×d
             * B = d_X+b*d (d×bs col-major = bs×d row-major), transb=N → d×bs
             * C = d_inner (n×bs col-major, ldc=n)
             * Result: C[j, i_batch] = -2 * dot(X[j], X[b+i_batch])
             * Kernel reads: inner[j + i_batch * n]
             */
            cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        n, bs, d,
                        &alpha, model->d_X, d,
                        model->d_X + b * d, d,
                        &beta_val, d_inner, n);

            int topk_grid = (bs + block - 1) / block;
            topk_from_gemm_kernel<<<topk_grid, block>>>(
                d_inner, d_norms, model->d_knn_indices, model->d_knn_distances,
                n, model->k, b, bs);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaMemcpy(model->h_knn_indices, model->d_knn_indices,
                          (size_t)n * (size_t)model->k * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(model->h_knn_distances, model->d_knn_distances,
                          (size_t)n * (size_t)model->k * sizeof(float), cudaMemcpyDeviceToHost));
    end_step = get_time_sec();
    log_printf("[DPC-AKNN]   -> Xong. (%.3f s)\n", end_step - start_step);

    /*
     * Buoc 2: d_c toan cuc theo Eq. (6)-(9). Du lieu dau vao chi con n*k.
     */
    log_printf("[DPC-AKNN] Buoc 2/8: Tinh d_c thich ung [HOST]...\n");
    start_step = get_time_sec();
    model->d_c = compute_dc_host(model->h_knn_distances, n, model->k);
    end_step = get_time_sec();
    log_printf("[DPC-AKNN]   -> d_c = %f (%.3f s)\n", model->d_c, end_step - start_step);

    /*
     * Buoc 3: rho va delta.
     */
    log_printf("[DPC-AKNN] Buoc 3/8a: Tinh rho [GPU]...\n");
    start_step = get_time_sec();
    compute_rho_kernel<<<grid, block>>>(model->d_knn_distances, model->d_rho, model->d_c, n, model->k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(model->h_rho, model->d_rho, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    end_step = get_time_sec();
    log_printf("[DPC-AKNN]   -> Xong. (%.3f s)\n", end_step - start_step);

    log_printf("[DPC-AKNN] Buoc 3/8b: Tinh delta via cuBLAS GEMM [GPU]...\n");
    start_step = get_time_sec();
    {
        int top = top_rho_index_host(model->h_rho, n);
        float alpha = -2.0f, beta_val = 0.0f;
        for (int b = 0; b < n; b += gemm_bs) {
            int bs = (b + gemm_bs <= n) ? gemm_bs : (n - b);
            cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        n, bs, d,
                        &alpha, model->d_X, d,
                        model->d_X + b * d, d,
                        &beta_val, d_inner, n);

            int delta_grid = (bs + block - 1) / block;
            delta_from_gemm_kernel<<<delta_grid, block>>>(
                d_inner, d_norms, model->d_rho, model->d_delta,
                n, b, bs, top);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaMemcpy(model->h_delta, model->d_delta, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    end_step = get_time_sec();
    log_printf("[DPC-AKNN]   -> Xong. (%.3f s)\n", end_step - start_step);

    /* Cleanup GEMM resources */
    cudaFree(d_inner);
    cudaFree(d_norms);
    cublasDestroy(cublas_handle);

    /*
     * Buoc 4: gamma = rho * delta, lay n_clusters diem lon nhat.
     */
    log_printf("[DPC-AKNN] Buoc 4/8: Chon %d tam cum [HOST]...\n", model->n_clusters);
    start_step = get_time_sec();
    select_centers_host(model);
    end_step = get_time_sec();
    log_printf("[DPC-AKNN]   -> Xong. (%.3f s)\n", end_step - start_step);

    /*
     * Buoc 5: cum nong cot ban dau, giu serial nhung update centroid tang dan.
     */
    log_printf("[DPC-AKNN] Buoc 5/8: Cum nong cot ban dau (BFS) [HOST]...\n");
    start_step = get_time_sec();
    build_initial_clusters_host(model, h_X);
    end_step = get_time_sec();
    log_printf("[DPC-AKNN]   -> Xong. (%.3f s)\n", end_step - start_step);

    /*
     * Buoc 6: ma tran lien ket A theo Eq. (10)-(11), chạy trên Host (CPU).
     */
    log_printf("[DPC-AKNN] Buoc 6/8: Ma tran lien ket A [HOST]...\n");
    start_step = get_time_sec();
    association_loop_cpu(model);
    end_step = get_time_sec();
    log_printf("[DPC-AKNN]   -> Xong. (%.3f s)\n", end_step - start_step);

    /*
     * Buoc 7 va 8: voting va vet can ngoai lai tren GPU.
     */
    log_printf("[DPC-AKNN] Buoc 7/8: Bau chon sua loi [GPU]...\n");
    start_step = get_time_sec();
    voting_gpu(model);
    end_step = get_time_sec();
    log_printf("[DPC-AKNN]   -> Xong. (%.3f s)\n", end_step - start_step);

    log_printf("[DPC-AKNN] Buoc 8/8: Vet can ngoai lai [GPU]...\n");
    start_step = get_time_sec();
    allocate_remaining_gpu(model);
    end_step = get_time_sec();
    log_printf("[DPC-AKNN]   -> Xong. (%.3f s)\n", end_step - start_step);
}

const int* dpcaknn_gpu_fit_predict(DPCAKNN_GPU* model, const float* h_X, int n, int d) {
    dpcaknn_gpu_fit(model, h_X, n, d);
    return model->h_labels;
}

void dpcaknn_gpu_save_labels(const DPCAKNN_GPU* model, const char* filepath) {
    csv_write_labels(filepath, model->h_labels, model->n);
}

void dpcaknn_gpu_free(DPCAKNN_GPU* model) {
    free(model->h_labels);
    free(model->h_centers);
    free(model->h_rho);
    free(model->h_delta);
    free(model->h_gamma);
    free(model->h_knn_indices);
    free(model->h_knn_distances);
    if (model->d_X) cudaFree(model->d_X);
    if (model->d_rho) cudaFree(model->d_rho);
    if (model->d_delta) cudaFree(model->d_delta);
    if (model->d_labels) cudaFree(model->d_labels);
    if (model->d_knn_indices) cudaFree(model->d_knn_indices);
    if (model->d_knn_distances) cudaFree(model->d_knn_distances);
    memset(model, 0, sizeof(*model));
}
