/* CUDA kernels used by the DPC-AKNN host orchestration layer. */
#ifndef KERNELS_CUH
#define KERNELS_CUH

/* CUDA keyword shims used only by IntelliSense. */
#ifdef __INTELLISENSE__
#define __global__
#define __host__
#define __device__
#define __shared__
#define __align__(n)
#define __launch_bounds__(t, b)
#define __device_builtin__
#define __cudart_builtin__
#define __noinline__
#define __forceinline__
#endif

__global__ void compute_knn_kernel(const float* X, int* knn_indices, float* knn_distances,
                                   int n, int d, int k);
__global__ void compute_rho_kernel(const float* knn_distances, float* rho, float d_c, int n, int k);
__global__ void compute_delta_kernel(const float* X, const float* rho, float* delta,
                                     int n, int d, int top_idx);
__global__ void build_initial_association_kernel(const int* knn_indices, const float* knn_distances,
                                              const int* labels, const int* active, const float* rho,
                                                 float* A, float* row_best_values, int* row_best_clusters,
                                                 int n, int k, int n_c);
__global__ void find_best_association_kernel(const float* row_best_values,
                                             const int* row_best_clusters,
                                             const int* active_points,
                                             float* block_values, int* block_indices,
                                             int remaining, int n_c);
__global__ void assign_point_kernel(int* labels, int* active, int* active_points,
                                    int best_pos, int remaining, int point, int cluster);
__global__ void update_association_column_kernel(const int* knn_indices, const float* knn_distances,
                                                 const int* active, const int* rknn_data,
                                                 const int* rknn_offset, const float* rho, float* A,
                                                 float* row_best_values, int* row_best_clusters,
                                                 int assigned_point, int assigned_cluster,
                                                 int k, int n_c);
__global__ void knn_voting_kernel(const int* knn_indices, const float* knn_distances, const int* labels,
                                  const int* order, int* new_labels, int n, int k, int n_c);
__global__ void allocate_remaining_kernel(const int* knn_indices, const float* knn_distances,
                                          int* labels, int n, int k, int n_c);

/* Kernels that consume batched cuBLAS GEMM output. */
__global__ void compute_norms_kernel(const float* X, float* norms, int n, int d);
__global__ void topk_from_gemm_kernel(const float* inner, const float* norms,
                                       int* knn_indices, float* knn_distances,
                                       int n, int k, int batch_start, int batch_size);
__global__ void delta_from_gemm_kernel(const float* inner, const float* norms,
                                        const float* rho, float* delta,
                                        int n, int batch_start, int batch_size, int top_idx);

#endif
