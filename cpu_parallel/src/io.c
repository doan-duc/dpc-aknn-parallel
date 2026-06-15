/* Serial CSV, output-directory, timestamp, and logging utilities. */
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

/* Internal helpers. */

static int count_columns(const char* line) {
    int cols = 1;
    for (const char* p = line; *p; p++)
        if (*p == ',') cols++;
    return cols;
}

/* Public API. */

real_t* io_read_matrix(const char* filepath, int* n_out, int* d_out) {
    /*
     * Empty lines are ignored. The first non-empty row determines the
     * matrix width. The caller owns the returned row-major buffer.
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
     * Non-integer rows, including an optional header, are ignored. The
     * caller owns the returned label buffer.
     */
    FILE* fp = fopen(filepath, "r");
    if (!fp) return NULL;

    char line[256];
    /* Count rows that contain a valid integer label. */
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
    /* Write a header followed by one integer label per row. */
    FILE* fp = fopen(filepath, "w");
    if (!fp) return;
    fprintf(fp, "label\n");
    for (int i = 0; i < n; i++) fprintf(fp, "%d\n", labels[i]);
    fclose(fp);
}

int io_ensure_output_dirs(void) {
    /* Existing-directory errors are intentionally ignored. */
    MKDIR(OUTPUT_DIR);
    MKDIR(LABELS_DIR);
    MKDIR(LOG_DIR);
    return 0;
}

const char* io_timestamp(void) {
    /* The static buffer is overwritten on each call and is not thread-safe. */
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
    
    /* Write to stdout. */
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    
    /* Mirror the message to the log file when logging is enabled. */
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
