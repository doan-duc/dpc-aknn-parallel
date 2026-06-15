/* CUDA error handling, CSV I/O, timing, metrics, and logging utilities. */
#ifndef UTILS_GPU_H
#define UTILS_GPU_H

#include "config.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <stdarg.h>
#include <stddef.h>

void gpu_check_error(cudaError_t err, const char* file, int line);
void* gpu_malloc_check(size_t bytes, const char* var_name);
int ensure_output_dirs(void);
const char* timestamp_str(void);
float* csv_read_matrix(const char* filepath, int* n_out, int* d_out);
void csv_write_labels(const char* filepath, const int* labels, int n);
int* csv_read_labels(const char* filepath, int n);
double get_time_sec(void);
double adjusted_rand_index(const int* y_true, const int* y_pred, int n);
double normalized_mutual_info(const int* y_true, const int* y_pred, int n);
double clustering_accuracy(const int* y_true, const int* y_pred, int n);
void log_printf(const char* format, ...);
extern FILE* gpu_log_fp;

#define CUDA_CHECK(call) gpu_check_error((call), __FILE__, __LINE__)

#endif
