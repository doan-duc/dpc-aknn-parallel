/* CSV, output-directory, timestamp, and logging utilities. */
#ifndef IO_H
#define IO_H

#include "config.h"

/* Read a CSV matrix in row-major order. The caller owns the result. */
real_t* io_read_matrix(const char* filepath, int* n_out, int* d_out);

/* Read a one-column integer label file. The caller owns the result. */
int* io_read_labels(const char* filepath, int* n_out);

/* Write cluster labels to a one-column CSV file named "label". */
void io_write_labels(const char* filepath, const int* labels, int n);

/* Create the output directories when they do not already exist. */
int io_ensure_output_dirs(void);

/* Return a timestamp in YYYYMMDD_HHMMSS format using a static buffer. */
const char* io_timestamp(void);

/* Open the log file used by io_log(). */
void io_init_logging(const char* log_filename);

/* Write a formatted message to stdout and the active log file. */
void io_log(const char* fmt, ...);

/* Close the active log file. */
void io_close_logging(void);

#endif /* IO_H */
