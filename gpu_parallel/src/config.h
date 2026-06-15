/* Configuration constants for the CUDA implementation of DPC-AKNN. */
#ifndef CONFIG_H
#define CONFIG_H

#define DATA_DIR "../data"
#define OUTPUT_DIR "output"
#define PLOTS_DIR "output/plots"
#define LABELS_DIR "output/labels"
#define LOG_DIR "output/logs"

/* These files are used when no command-line input is provided. */
#define DEFAULT_INPUT_FILE "../data/real/fashion-mnist/fashion_mnist_X.csv"
#define DEFAULT_LABEL_FILE "../data/real/fashion-mnist/fashion_mnist_y.csv"

#define DEFAULT_N_CLUSTERS 10
#define DEFAULT_K 15

#define GPU_DEVICE_ID 0
#define TILE_SIZE 16
#define BLOCK_SIZE_1D 256
#define BLOCK_SIZE_2D_X 16
#define BLOCK_SIZE_2D_Y 16
/*
 * Number of input rows processed by each GEMM batch. Choose a value that fits
 * the available device memory; the configured value is used without scaling.
 */
#define GPU_BATCH_SIZE 1000
#define GPU_FALLBACK_CPU 1
#define GPU_MAX_K 64
#define GPU_MAX_REDUCTION_BLOCKS 4096

typedef float real_t;

#define LOG_LEVEL 1
#define EPS_DISTANCE 1e-7f

#endif
