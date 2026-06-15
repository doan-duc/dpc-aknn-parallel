/* Host and device state for the CUDA DPC-AKNN implementation. */
#ifndef DPC_AKNN_GPU_H
#define DPC_AKNN_GPU_H

#include "config.h"

typedef struct {
    int n_clusters;
    int k;
    int n;
    int d;
    int* h_labels;
    int* h_centers;
    float* h_rho;
    float* h_delta;
    float* h_gamma;
    float d_c;
    int* h_knn_indices;
    float* h_knn_distances;
    float* d_X;
    float* d_rho;
    float* d_delta;
    int* d_labels;
    int* d_knn_indices;
    float* d_knn_distances;
} DPCAKNN_GPU;

void dpcaknn_gpu_init(DPCAKNN_GPU* model, int n_clusters, int k);
void dpcaknn_gpu_fit(DPCAKNN_GPU* model, const float* h_X, int n, int d);
const int* dpcaknn_gpu_fit_predict(DPCAKNN_GPU* model, const float* h_X, int n, int d);
void dpcaknn_gpu_save_labels(const DPCAKNN_GPU* model, const char* filepath);
void dpcaknn_gpu_free(DPCAKNN_GPU* model);

float compute_dc_host(const float* knn_distances, int n, int k);

#endif
