/*
 * Configuration constants for the C99 and OpenMP implementation of
 * DPC-AKNN.
 */
#ifndef CONFIG_H
#define CONFIG_H

/* I/O paths. */
#define DATA_DIR        "../data"
#define OUTPUT_DIR      "output"
#define LABELS_DIR      "output/labels"
#define LOG_DIR         "output/logs"

/* Default input files and algorithm parameters. */
/* These paths are used when no command-line input is provided. */
#define DEFAULT_INPUT_FILE  "../data/real/iris/iris_X_norm.csv"
#define DEFAULT_LABEL_FILE  "../data/real/iris/iris_y.csv"

#define DEFAULT_N_CLUSTERS  3
#define DEFAULT_K           7

/* OpenMP configuration. Set to 0 to use omp_get_max_threads(). */
#define OMP_N_THREADS   16
#define OMP_CHUNK_SIZE  64

/* Floating-point type used by the CPU implementation. */
typedef double real_t;

/* Lower bound used to avoid division by zero. */
#define EPS_DISTANCE    1e-10

/* Logging level: 0 = off, 1 = information, 2 = debug. */
#define LOG_LEVEL       1

#endif /* CONFIG_H */
