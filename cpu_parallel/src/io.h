/*
 * io.h - Khai báo tiện ích I/O cho bản CPU Parallel.
 *
 * Mục đích: Cung cấp đọc/ghi CSV và tạo thư mục output nhất quán.
 * Song song hóa: Không song song.
 */
#ifndef IO_H
#define IO_H

#include "config.h"

/* Đọc file CSV thành ma trận real_t row-major. Caller phải free(). */
real_t* io_read_matrix(const char* filepath, int* n_out, int* d_out);

/* Đọc file CSV nhãn nguyên (1 cột). Caller phải free(). */
int* io_read_labels(const char* filepath, int* n_out);

/* Ghi nhãn phân cụm ra file CSV 1 cột "label". */
void io_write_labels(const char* filepath, const int* labels, int n);

/* Tạo các thư mục output nếu chưa tồn tại. Trả về 0 nếu thành công. */
int io_ensure_output_dirs(void);

/* Trả về chuỗi timestamp dạng "YYYYMMDD_HHMMSS" (static buffer). */
const char* io_timestamp(void);

/* Khởi tạo ghi log vào file (gọi ở đầu main) */
void io_init_logging(const char* log_filename);

/* In log ra cả stdout và file (nếu đã khởi tạo) */
void io_log(const char* fmt, ...);

/* Đóng file log */
void io_close_logging(void);

#endif /* IO_H */
