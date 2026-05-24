/*
 * main.cu - Entry point cho gpu_parallel.
 *
 * Muc dich: Phan cum file CSV do nguoi dung cung cap.
 * Bài báo: Thực nghiệm DPC-AKNN.
 * Song song hóa: Đo thời gian bằng CUDA Events.
 */
#include "dpc_aknn.h"
#include "utils_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void run_input(const char* filepath, const char* labelpath, int n_clusters, int k_val) {
    int n = 0, d = 0;
    
    char log_path[512];
    snprintf(log_path, sizeof(log_path), LOG_DIR "/gpu_run_%s.txt", timestamp_str());
    gpu_log_fp = fopen(log_path, "w");

    log_printf("[Config] GPU Device ID: %d\n", GPU_DEVICE_ID);
    log_printf("[I/O] Dang doc du lieu: %s ...\n", filepath);
    float* X = csv_read_matrix(filepath, &n, &d);
    if (!X) {
        fprintf(stderr, "[Loi] Khong doc duoc file: %s\n", filepath);
        if (gpu_log_fp) fclose(gpu_log_fp);
        return;
    }
    log_printf("[I/O] Da doc: n=%d mau, d=%d chieu\n", n, d);

    int* y_true = NULL;
    if (labelpath) {
        y_true = csv_read_labels(labelpath, n);
    }

    log_printf("[Run] Bat dau chay DPC-AKNN (GPU)...\n");
    double start_total = get_time_sec();

    DPCAKNN_GPU model;
    dpcaknn_gpu_init(&model, n_clusters, k_val);
    dpcaknn_gpu_fit(&model, X, n, d);

    double end_total = get_time_sec();
    double total_time = end_total - start_total;

    char out_path[512];
    snprintf(out_path, sizeof(out_path), LABELS_DIR "/gpu_labels_%s.csv", timestamp_str());
    dpcaknn_gpu_save_labels(&model, out_path);

    log_printf("\n========================================\n");
    log_printf("  KET QUA PHAN CUM - DPC-AKNN (GPU)\n");
    log_printf("========================================\n");
    log_printf("  So mau      : %d\n", n);
    log_printf("  So chieu    : %d\n", d);
    log_printf("  So cum      : %d\n", n_clusters);
    log_printf("  K lang gieng: %d\n", k_val);
    log_printf("  Tong thoi gian: %.4f giay\n", total_time);
    log_printf("  Ket qua luu : %s\n", out_path);
    log_printf("  Log file    : %s\n", log_path);

    if (y_true) {
        double ari = adjusted_rand_index(y_true, model.h_labels, n);
        double nmi = normalized_mutual_info(y_true, model.h_labels, n);
        double acc = clustering_accuracy(y_true, model.h_labels, n);
        log_printf("\n  --- Cac chi so danh gia ---\n");
        log_printf("  ARI (Adjusted Rand Index)  : %.4f\n", ari);
        log_printf("  NMI (Normalized Mutual Info): %.4f\n", nmi);
        log_printf("  ACC (Clustering Accuracy)   : %.4f\n", acc);
        free(y_true);
    }
    log_printf("========================================\n\n");

    if (gpu_log_fp) {
        fclose(gpu_log_fp);
        gpu_log_fp = NULL;
    }
    dpcaknn_gpu_free(&model);
    free(X);
}

int main(int argc, char** argv) {
    ensure_output_dirs();
    const char* input_file = NULL;
    const char* label_file = NULL;
    int n_clusters = DEFAULT_N_CLUSTERS;
    int k_val = DEFAULT_K;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) input_file = argv[++i];
        else if (strcmp(argv[i], "--labels") == 0 && i + 1 < argc) label_file = argv[++i];
        else if (strcmp(argv[i], "--clusters") == 0 && i + 1 < argc) n_clusters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) k_val = atoi(argv[++i]);
    }

    if (!input_file) {
        input_file = DEFAULT_INPUT_FILE;
        if (!label_file) {
            label_file = DEFAULT_LABEL_FILE;
        }
        printf("[System] Khong co doi so dau vao. Su dung cau hinh mac dinh trong config.h:\n");
        printf("         Data    : %s\n", input_file);
        if (label_file) {
            printf("         Labels  : %s\n", label_file);
        }
        printf("         Clusters: %d, K: %d\n\n", n_clusters, k_val);
    }

    run_input(input_file, label_file, n_clusters, k_val);
    return 0;
}
