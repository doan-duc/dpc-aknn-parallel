/* CUDA error handling, CSV I/O, timing, metrics, and logging utilities. */
#include "utils_gpu.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#define MKDIR(path) mkdir(path, 0755)
#endif

FILE* gpu_log_fp = NULL;

void log_printf(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);

    if (gpu_log_fp) {
        va_start(args, format);
        vfprintf(gpu_log_fp, format, args);
        va_end(args);
        fflush(gpu_log_fp);
    }
}

void gpu_check_error(cudaError_t err, const char* file, int line) {
    /* Abort immediately when a CUDA runtime operation fails. */
    if (err != cudaSuccess) {
        const char* err_str = cudaGetErrorString(err);
        if (!err_str) err_str = "unknown CUDA error";
        fprintf(stderr, "Loi CUDA tai %s:%d: %s (code=%d)\n", file, line, err_str, (int)err);
        exit(1);
    }
}

void* gpu_malloc_check(size_t bytes, const char* var_name) {
    /* Allocate device memory and report the requested buffer on failure. */
    void* ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Khong cap phat duoc %s (%zu bytes): %s\n", var_name, bytes, cudaGetErrorString(err));
        exit(1);
    }
    return ptr;
}

int ensure_output_dirs(void) {
    MKDIR(OUTPUT_DIR);
    MKDIR(PLOTS_DIR);
    MKDIR(LABELS_DIR);
    MKDIR(LOG_DIR);
    return 0;
}

const char* timestamp_str(void) {
    static char buffer[32];
    time_t now = time(NULL);
    struct tm* tm_now = localtime(&now);
    strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", tm_now);
    return buffer;
}

static int count_columns(const char* line) {
    int cols = 1;
    for (const char* p = line; *p; p++) if (*p == ',') cols++;
    return cols;
}

float* csv_read_matrix(const char* filepath, int* n_out, int* d_out) {
    FILE* fp = fopen(filepath, "r");
    if (!fp) return NULL;
    char line[8192];
    int n = 0, d = 0;
    while (fgets(line, sizeof(line), fp)) {
        char* p = line;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '\0') continue;
        if (d == 0) d = count_columns(line);
        n++;
    }
    rewind(fp);
    float* X = (float*)malloc((size_t)n * (size_t)d * sizeof(float));
    int row = 0;
    while (fgets(line, sizeof(line), fp) && row < n) {
        char* cursor = line;
        for (int col = 0; col < d; col++) {
            X[row * d + col] = strtof(cursor, &cursor);
            if (*cursor == ',') cursor++;
        }
        row++;
    }
    fclose(fp);
    *n_out = n;
    *d_out = d;
    return X;
}

void csv_write_labels(const char* filepath, const int* labels, int n) {
    FILE* fp = fopen(filepath, "w");
    if (!fp) return;
    fprintf(fp, "label\n");
    for (int i = 0; i < n; i++) fprintf(fp, "%d\n", labels[i]);
    fclose(fp);
}

int* csv_read_labels(const char* filepath, int n) {
    FILE* fp = fopen(filepath, "r");
    if (!fp) return NULL;
    char line[1024];
    
    /* Read the first row to detect an optional header. */
    if (!fgets(line, sizeof(line), fp)) { fclose(fp); return NULL; }
    
    int* labels = (int*)malloc((size_t)n * sizeof(int));
    int start_idx = 0;
    
    /* A non-numeric first character indicates a header row. */
    if (!isdigit((unsigned char)line[0]) && line[0] != '-' && line[0] != ' ') {
        /* Skip the header; the next read starts the label data. */
    } else {
        /* Preserve a numeric first row as the first label. */
        labels[0] = atoi(line);
        start_idx = 1;
    }
    
    for (int i = start_idx; i < n; i++) {
        if (!fgets(line, sizeof(line), fp)) labels[i] = -1;
        else labels[i] = atoi(line);
    }
    fclose(fp);
    return labels;
}

double get_time_sec(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
#endif
}

static double comb2(int x) {
    return x < 2 ? 0.0 : (double)x * (double)(x - 1) / 2.0;
}

double adjusted_rand_index(const int* y_true, const int* y_pred, int n) {
    int max_true = 0, max_pred = 0;
    int valid_n = 0;
    for (int i = 0; i < n; i++) {
        if (y_true[i] < 0 || y_pred[i] < 0) continue;
        if (y_true[i] > max_true) max_true = y_true[i];
        if (y_pred[i] > max_pred) max_pred = y_pred[i];
        valid_n++;
    }
    
    if (valid_n == 0) return 0.0;
    
    int rows = max_true + 1, cols = max_pred + 1;
    int* table = (int*)calloc((size_t)rows * (size_t)cols, sizeof(int));
    int* rs = (int*)calloc((size_t)rows, sizeof(int));
    int* cs = (int*)calloc((size_t)cols, sizeof(int));
    for (int i = 0; i < n; i++) {
        if (y_true[i] < 0 || y_pred[i] < 0) continue;
        table[y_true[i] * cols + y_pred[i]]++;
        rs[y_true[i]]++;
        cs[y_pred[i]]++;
    }
    double nij = 0.0, ai = 0.0, bj = 0.0;
    for (int r = 0; r < rows; r++) {
        ai += comb2(rs[r]);
        for (int c = 0; c < cols; c++) nij += comb2(table[r * cols + c]);
    }
    for (int c = 0; c < cols; c++) bj += comb2(cs[c]);
    double total = comb2(valid_n);
    double expected = total > 0.0 ? ai * bj / total : 0.0;
    double max_index = 0.5 * (ai + bj);
    free(table);
    free(rs);
    free(cs);
    if (fabs(max_index - expected) < 1e-12) return 1.0;
    return (nij - expected) / (max_index - expected);
}

double normalized_mutual_info(const int* y_true, const int* y_pred, int n) {
    int max_true = 0, max_pred = 0;
    int valid_n = 0;
    for (int i = 0; i < n; i++) {
        if (y_true[i] < 0 || y_pred[i] < 0) continue;
        if (y_true[i] > max_true) max_true = y_true[i];
        if (y_pred[i] > max_pred) max_pred = y_pred[i];
        valid_n++;
    }
    if (valid_n == 0) return 0.0;

    int rows = max_true + 1, cols = max_pred + 1;
    int* table = (int*)calloc((size_t)rows * (size_t)cols, sizeof(int));
    int* rs = (int*)calloc((size_t)rows, sizeof(int));
    int* cs = (int*)calloc((size_t)cols, sizeof(int));
    for (int i = 0; i < n; i++) {
        if (y_true[i] < 0 || y_pred[i] < 0) continue;
        table[y_true[i] * cols + y_pred[i]]++;
        rs[y_true[i]]++;
        cs[y_pred[i]]++;
    }

    double mi = 0.0;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (table[r * cols + c] > 0) {
                double p_xy = (double)table[r * cols + c] / valid_n;
                double p_x = (double)rs[r] / valid_n;
                double p_y = (double)cs[c] / valid_n;
                mi += p_xy * log(p_xy / (p_x * p_y));
            }
        }
    }

    double hx = 0.0, hy = 0.0;
    for (int r = 0; r < rows; r++) {
        if (rs[r] > 0) {
            double p = (double)rs[r] / valid_n;
            hx -= p * log(p);
        }
    }
    for (int c = 0; c < cols; c++) {
        if (cs[c] > 0) {
            double p = (double)cs[c] / valid_n;
            hy -= p * log(p);
        }
    }

    free(table); free(rs); free(cs);
    if (hx + hy == 0.0) return 1.0;
    return 2.0 * mi / (hx + hy);
}

double clustering_accuracy(const int* y_true, const int* y_pred, int n) {
    int max_true = 0, max_pred = 0;
    for (int i = 0; i < n; i++) {
        if (y_true[i] < 0 || y_pred[i] < 0) continue;
        if (y_true[i] > max_true) max_true = y_true[i];
        if (y_pred[i] > max_pred) max_pred = y_pred[i];
    }
    int rows = max_true + 1, cols = max_pred + 1;
    int* table = (int*)calloc((size_t)rows * (size_t)cols, sizeof(int));
    for (int i = 0; i < n; i++) {
        if (y_true[i] < 0 || y_pred[i] < 0) continue;
        table[y_true[i] * cols + y_pred[i]]++;
    }

    /* Approximate clustering accuracy with greedy label matching. */
    int* matched_pred = (int*)malloc((size_t)cols * sizeof(int));
    int* label_taken = (int*)calloc((size_t)rows, sizeof(int));
    for (int c = 0; c < cols; c++) matched_pred[c] = -1;

    for (int iter = 0; iter < cols; iter++) {
        int best_r = -1, best_c = -1, max_v = -1;
        for (int c = 0; c < cols; c++) {
            if (matched_pred[c] != -1) continue;
            for (int r = 0; r < rows; r++) {
                if (label_taken[r]) continue;
                if (table[r * cols + c] > max_v) {
                    max_v = table[r * cols + c];
                    best_r = r;
                    best_c = c;
                }
            }
        }
        if (best_r != -1) {
            matched_pred[best_c] = best_r;
            label_taken[best_r] = 1;
        } else break;
    }

    long long correct = 0;
    long long valid_n = 0;
    for (int i = 0; i < n; i++) {
        if (y_true[i] < 0 || y_pred[i] < 0) continue;
        valid_n++;
        if (matched_pred[y_pred[i]] == y_true[i]) correct++;
    }

    free(table); free(matched_pred); free(label_taken);
    return valid_n > 0 ? (double)correct / valid_n : 0.0;
}
