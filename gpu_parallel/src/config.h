/*
 * config.h - Cấu hình cho phiên bản GPU PARALLEL (CUDA C).
 *
 * Mục đích: Giữ tham số khớp với original/ và cpu_parallel/.
 * Bài báo: Dùng cho Algorithm 1 của DPC-AKNN.
 * Song song hóa: Không song song.
 */
#ifndef CONFIG_H
#define CONFIG_H

#define DATA_DIR "../data"
#define OUTPUT_DIR "output"
#define PLOTS_DIR "output/plots"
#define LABELS_DIR "output/labels"
#define LOG_DIR "output/logs"

// Đường dẫn mặc định của file dữ liệu và nhãn (có thể sửa ở đây để chạy không cần đối số dòng lệnh)
#define DEFAULT_INPUT_FILE "../data/real/fashion-mnist/fashion_mnist_X.csv"
#define DEFAULT_LABEL_FILE "../data/real/fashion-mnist/fashion_mnist_y.csv"

#define DEFAULT_N_CLUSTERS 10
#define DEFAULT_K 15

#define GPU_DEVICE_ID 0
#define TILE_SIZE 16
#define BLOCK_SIZE_1D 256
#define BLOCK_SIZE_2D_X 16
#define BLOCK_SIZE_2D_Y 16
// Kích thước phân đoạn (Batch Size) cho GPU. Bạn có thể tự do tăng/giảm giá trị này tùy ý theo dung lượng VRAM.
// Code đã được loại bỏ giới hạn tự động giảm, giá trị bạn điền ở đây sẽ được áp dụng chính xác 100%.
#define GPU_BATCH_SIZE 1000
#define GPU_FALLBACK_CPU 1
#define GPU_MAX_K 64
#define GPU_MAX_REDUCTION_BLOCKS 4096

typedef float real_t;

#define LOG_LEVEL 1
#define EPS_DISTANCE 1e-7f

#endif
