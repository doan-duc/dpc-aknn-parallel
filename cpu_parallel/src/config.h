/*
 * config.h - Cấu hình cho phiên bản CPU PARALLEL (C99 + OpenMP).
 *
 * Mục đích: Giữ tham số thuật toán khớp với bản Python và CUDA.
 * Song song hóa: Không song song.
 */
#ifndef CONFIG_H
#define CONFIG_H

/* ─── Đường dẫn I/O ─────────────────────────────────────────────────────── */
#define DATA_DIR        "../data"
#define OUTPUT_DIR      "output"
#define LABELS_DIR      "output/labels"
#define LOG_DIR         "output/logs"

/* ─── Tham số thuật toán mặc định ──────────────────────────────────────── */
#define DEFAULT_N_CLUSTERS  10
#define DEFAULT_K           7

/* ─── Cấu hình OpenMP ──────────────────────────────────────────────────── */
/*  0 = dùng tất cả lõi khả dụng (omp_get_max_threads)                     */
#define OMP_N_THREADS   16
#define OMP_CHUNK_SIZE  64

/* ─── Kiểu dữ liệu dấu phẩy động ──────────────────────────────────────── */
typedef double real_t;

/* ─── Hằng số bảo vệ chia-cho-không ────────────────────────────────────── */
#define EPS_DISTANCE    1e-10

/* ─── Mức log: 0=tắt, 1=info, 2=debug ──────────────────────────────────── */
#define LOG_LEVEL       1

#endif /* CONFIG_H */
