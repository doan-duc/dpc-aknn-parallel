/*
 * io.c - Triển khai tiện ích I/O cho bản CPU Parallel.
 *
 * Mục đích: Đọc dữ liệu CSV, ghi nhãn phân cụm, tạo thư mục output.
 * Song song hóa: Không song song (I/O tuần tự theo thiết kế).
 */
#include "io.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>

#ifdef _WIN32
#  include <direct.h>
#  define MKDIR(p) _mkdir(p)
#else
#  include <sys/stat.h>
#  define MKDIR(p) mkdir((p), 0755)
#endif

/* ─── Hàm nội bộ ───────────────────────────────────────────────────────── */

static int count_columns(const char* line) {
    int cols = 1;
    for (const char* p = line; *p; p++)
        if (*p == ',') cols++;
    return cols;
}

/* ─── API công khai ─────────────────────────────────────────────────────── */

real_t* io_read_matrix(const char* filepath, int* n_out, int* d_out) {
    /*
     * io_read_matrix - Đọc file CSV thành ma trận real_t.
     *
     * Mục đích: Nạp dữ liệu đầu vào dạng row-major trước Bước 1 (kNN).
     * Lưu ý: Bỏ qua các dòng trống; cột đầu tiên quyết định số chiều d.
     * Caller có trách nhiệm free() con trỏ trả về.
     */
    FILE* fp = fopen(filepath, "r");
    if (!fp) return NULL;

    char line[65536];
    int n = 0, d = 0;
    while (fgets(line, sizeof(line), fp)) {
        char* p = line;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '\0') continue;
        if (d == 0) d = count_columns(line);
        n++;
    }
    rewind(fp);

    real_t* X = (real_t*)malloc((size_t)n * (size_t)d * sizeof(real_t));
    if (!X) { fclose(fp); return NULL; }

    int row = 0;
    while (fgets(line, sizeof(line), fp) && row < n) {
        char* cursor = line;
        for (int col = 0; col < d; col++) {
            X[row * d + col] = strtod(cursor, &cursor);
            if (*cursor == ',') cursor++;
        }
        row++;
    }
    fclose(fp);
    *n_out = n;
    *d_out = d;
    return X;
}

int* io_read_labels(const char* filepath, int* n_out) {
    /*
     * io_read_labels - Đọc file nhãn CSV 1 cột vào mảng int.
     *
     * Mục đích: Nạp nhãn ground-truth để tính ARI, NMI, ACC.
     * Lưu ý: Bỏ qua dòng header nếu không parse được thành số nguyên.
     * Caller có trách nhiệm free() con trỏ trả về.
     */
    FILE* fp = fopen(filepath, "r");
    if (!fp) return NULL;

    char line[256];
    /* Đếm số dòng số */
    int n = 0;
    while (fgets(line, sizeof(line), fp)) {
        char* end;
        strtol(line, &end, 10);
        if (end != line && (*end == '\n' || *end == '\r' || *end == '\0')) n++;
    }
    rewind(fp);

    int* labels = (int*)malloc((size_t)n * sizeof(int));
    if (!labels) { fclose(fp); return NULL; }

    int row = 0;
    while (fgets(line, sizeof(line), fp) && row < n) {
        char* end;
        long val = strtol(line, &end, 10);
        if (end != line && (*end == '\n' || *end == '\r' || *end == '\0'))
            labels[row++] = (int)val;
    }
    fclose(fp);
    *n_out = n;
    return labels;
}

void io_write_labels(const char* filepath, const int* labels, int n) {
    /*
     * io_write_labels - Ghi nhãn phân cụm ra file CSV 1 cột.
     *
     * Mục đích: Lưu kết quả để Python đọc tính metric và vẽ biểu đồ.
     * Định dạng: header "label", mỗi dòng là 1 số nguyên.
     */
    FILE* fp = fopen(filepath, "w");
    if (!fp) return;
    fprintf(fp, "label\n");
    for (int i = 0; i < n; i++) fprintf(fp, "%d\n", labels[i]);
    fclose(fp);
}

int io_ensure_output_dirs(void) {
    /*
     * io_ensure_output_dirs - Tạo toàn bộ thư mục output.
     *
     * Mục đích: Tránh lỗi "file not found" khi ghi kết quả.
     * Ghi chú: MKDIR() trả về lỗi nếu thư mục đã tồn tại — bỏ qua điều đó.
     */
    MKDIR(OUTPUT_DIR);
    MKDIR(LABELS_DIR);
    MKDIR(LOG_DIR);
    return 0;
}

const char* io_timestamp(void) {
    /*
     * io_timestamp - Tạo chuỗi thời gian cho tên file output.
     *
     * Mục đích: Tránh ghi đè kết quả giữa các lần chạy.
     * Lưu ý: Dùng static buffer — không thread-safe.
     */
    static char buffer[32];
    time_t now = time(NULL);
    struct tm* tm_now = localtime(&now);
    strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", tm_now);
    return buffer;
}

static FILE* g_log_file = NULL;

void io_init_logging(const char* log_filename) {
    if (g_log_file) {
        fclose(g_log_file);
    }
    g_log_file = fopen(log_filename, "w");
    if (!g_log_file) {
        fprintf(stderr, "[Loi] Khong the mo file log de ghi: %s\n", log_filename);
    }
}

void io_log(const char* fmt, ...) {
    va_list args;
    
    // In ra stdout
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    
    // In ra file log
    if (g_log_file) {
        va_start(args, fmt);
        vfprintf(g_log_file, fmt, args);
        va_end(args);
        fflush(g_log_file);
    }
}

void io_close_logging(void) {
    if (g_log_file) {
        fclose(g_log_file);
        g_log_file = NULL;
    }
}
