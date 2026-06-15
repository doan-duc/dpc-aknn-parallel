/*
 * CPU entry point for DPC-AKNN.
 *
 * Usage:
 *   ./dpc_aknn_cpu --input <X.csv> [--labels <y.csv>]
 *                  [--clusters <n>] [--k <k>]
 *
 * --input specifies the input matrix.
 * --labels optionally enables ARI, NMI, and ACC evaluation.
 * --clusters and --k override the defaults in config.h.
 */
#include "dpc_aknn_algo.h"
#include "io.h"
#include "metrics.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
static double wall_time(void) { return omp_get_wtime(); }
#else
#include <time.h>
static double wall_time(void) { return (double)clock() / CLOCKS_PER_SEC; }
#endif



/* Program entry point. */

int main(int argc, char** argv) {
    /* Parse command-line arguments. */
    const char* input_path  = NULL;
    const char* labels_path = NULL;
    int n_clusters = DEFAULT_N_CLUSTERS;
    int k_val      = DEFAULT_K;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input")    == 0 && i+1 < argc) input_path  = argv[++i];
        else if (strcmp(argv[i], "--labels") == 0 && i+1 < argc) labels_path = argv[++i];
        else if (strcmp(argv[i], "--clusters") == 0 && i+1 < argc) n_clusters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--k")   == 0 && i+1 < argc) k_val      = atoi(argv[++i]);
    }

    if (!input_path) {
        input_path = DEFAULT_INPUT_FILE;
        if (!labels_path) {
            labels_path = DEFAULT_LABEL_FILE;
        }
        printf("[System] Khong co doi so dau vao. Su dung cau hinh mac dinh trong config.h:\n");
        printf("         Data    : %s\n", input_path);
        if (labels_path) {
            printf("         Labels  : %s\n", labels_path);
        }
        printf("         Clusters: %d, K: %d\n\n", n_clusters, k_val);
    }

    io_ensure_output_dirs();
    
    char log_path[512];
    snprintf(log_path, sizeof(log_path), LOG_DIR "/cpu_log_%s.txt", io_timestamp());
    io_init_logging(log_path);

#ifdef _OPENMP
    if (OMP_N_THREADS > 0) omp_set_num_threads(OMP_N_THREADS);
    io_log("[Config] OpenMP: %d luong\n", omp_get_max_threads());
#else
    io_log("[Config] OpenMP: khong co (bien dich don luong)\n");
#endif

    /* Load the input matrix. */
    io_log("[I/O] Dang doc du lieu: %s ...\n", input_path);
    int n = 0, d = 0;
    real_t* X = io_read_matrix(input_path, &n, &d);
    if (!X) { io_log("[Loi] khong doc duoc %s\n", input_path); io_close_logging(); return 1; }
    io_log("[I/O] Da doc: n=%d mau, d=%d chieu\n", n, d);

    /* Initialize and fit the model. */
    DPCAKNNModel model;
    algo_init(&model, n_clusters, k_val);

    io_log("[Run] Bat dau chay DPC-AKNN...\n");
    double t_start = wall_time();
    algo_fit(&model, X, n, d);
    double t_total = wall_time() - t_start;

    /* Save the predicted labels. */
    char out_path[512];
    snprintf(out_path, sizeof(out_path),
             LABELS_DIR "/cpu_labels_%s.csv", io_timestamp());
    algo_save_labels(&model, out_path);

    /* Report the run summary. */
    io_log("\n========================================\n");
    io_log("  KET QUA PHAN CUM - DPC-AKNN (CPU)\n");
    io_log("========================================\n");
    io_log("  So mau      : %d\n", n);
    io_log("  So chieu    : %d\n", d);
    io_log("  So cum      : %d\n", n_clusters);
    io_log("  K lang gieng: %d\n", k_val);
    io_log("  Tong thoi gian: %.4f giay\n", t_total);
    io_log("  Ket qua luu : %s\n", out_path);

    /* Evaluate the result when ground-truth labels are available. */
    if (labels_path) {
        int n_gt = 0;
        int* y_true = io_read_labels(labels_path, &n_gt);
        if (!y_true || n_gt != n) {
            io_log("  [Canh bao] Khong the doc labels hoac so luong khong khop.\n");
        } else {
            double ari = compute_ari(y_true, model.labels, n);
            double nmi = compute_nmi(y_true, model.labels, n);
            double acc = compute_acc(y_true, model.labels, n);
            io_log("\n  --- Cac chi so danh gia ---\n");
            io_log("  ARI (Adjusted Rand Index)  : %.4f\n", ari);
            io_log("  NMI (Normalized Mutual Info): %.4f\n", nmi);
            io_log("  ACC (Clustering Accuracy)   : %.4f\n", acc);
        }
        free(y_true);
    } else {
        io_log("\n  [Goi y] Them --labels <y.csv> de xem chi so ARI/NMI/ACC.\n");
    }
    io_log("========================================\n\n");

    algo_free(&model);
    free(X);
    io_close_logging();
    return 0;
}
