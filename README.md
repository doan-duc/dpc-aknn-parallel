# 🚀 Dự Án Song Song Hóa Thuật Toán Phân Cụm DPC-AKNN (OpenMP & CUDA)

[![C99](https://img.shields.io/badge/Language-C99-blue.svg)](https://en.wikipedia.org/wiki/C99)
[![CUDA](https://img.shields.io/badge/Parallel-CUDA_C-green.svg)](https://developer.nvidia.com/cuda-zone)
[![OpenMP](https://img.shields.io/badge/Parallel-OpenMP-orange.svg)](https://www.openmp.org/)
[![Python](https://img.shields.io/badge/Scripting-Python_3-yellow.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)

Dự án này tập trung triển khai và tối ưu hóa song song thuật toán **Density Peak Clustering based on Nearest Neighbors (DPC-AKNN)** trên cả CPU (sử dụng **OpenMP**) và GPU (sử dụng **CUDA C**). Thuật toán này được cải tiến dựa trên bài báo khoa học:

> 📄 **Houshen Lin, Jian Hou, Huaqiang Yuan.** *"Density peak clustering based on nearest neighbors"*. Engineering Applications of Artificial Intelligence 160 (2025) 111981.

---

## 📌 Tổng Quan Thuật Toán DPC-AKNN

DPC-AKNN khắc phục hai điểm hạn chế lớn nhất của thuật toán DPC (Density Peak Clustering) truyền thống bằng cách:
1. **Hàm mật độ cục bộ thích ứng ($\rho_i$):** Thay vì dùng ngưỡng cắt tĩnh (cutoff distance) trên toàn bộ không gian, DPC-AKNN ràng buộc việc tính mật độ cục bộ trong phạm vi $k$-lân cận gần nhất (kNN) với nhân Gauss (Gaussian kernel). Điều này giúp giảm nhiễu từ các điểm ở quá xa.
2. **Cơ chế gán nhãn lặp qua Ma trận Liên kết (Linkage Matrix):** Loại bỏ cơ chế gán nhãn một bước (dễ tích tụ sai số). DPC-AKNN xây dựng một Ma trận Liên kết $A$ chia sẻ giữa các cụm, thực hiện cập nhật lặp và hiệu chỉnh lại nhãn thông qua cơ chế bỏ phiếu kNN (kNN Voting).

Do độ phức tạp của thuật toán chủ yếu tập trung ở vòng lặp cập nhật Ma trận Liên kết ($O(N_c \times N^2)$), việc ứng dụng **OpenMP** và **CUDA** mang lại hiệu năng xử lý vượt trội, cho phép thuật toán chạy trên các tập dữ liệu lớn lên tới hàng chục vạn mẫu (như Fashion-MNIST với 70,000 ảnh).

---

## 📂 Cấu Trúc Thư Mục Dự Án

```text
D:\BTL_LTSS/
├── cpu_parallel/                     # Phiên bản song song hóa CPU (C99 + OpenMP)
│   ├── src/                          # Mã nguồn C (.c, .h)
│   │   ├── config.h                  # File cấu hình tham số mặc định & OpenMP
│   │   ├── main.c                    # Điểm chạy chính của chương trình
│   │   ├── dpc_aknn_core.c/.h        # Logic lõi của thuật toán DPC-AKNN
│   │   └── ...
│   ├── utility_functions/            # Công cụ hỗ trợ CPU
│   │   └── visualize.py              # Script so sánh kết quả CPU vs Ground-Truth
│   ├── Makefile                      # Trình biên dịch dự án CPU
│   ├── readme_win.md                 # Hướng dẫn chi tiết biên dịch trên Windows
│   └── readme_linux.md               # Hướng dẫn chi tiết biên dịch trên Linux
│
├── gpu_parallel/                     # Phiên bản song song hóa GPU (CUDA C)
│   ├── src/                          # Mã nguồn CUDA (.cu, .h, .cuh)
│   │   ├── config.h                  # File cấu hình tham số GPU (Batch size, Thread Block)
│   │   ├── main.cu                   # Điểm chạy chính của chương trình GPU
│   │   ├── kernels.cu/.cuh           # Các CUDA Kernels tính toán song song kNN, Linkage
│   │   └── ...
│   ├── utility_functions/            # Công cụ hỗ trợ GPU
│   │   ├── demo.py                   # Bản DEMO Python trực quan hóa động (Live-plot)
│   │   ├── metrics.py                # Tính toán ARI, NMI, ACC (Hungarian Algorithm)
│   │   └── visualize.py              # Vẽ biểu đồ so sánh phân cụm GPU
│   ├── Makefile                      # Trình biên dịch dự án GPU
│   ├── readme_win.md                 # Hướng dẫn chi tiết biên dịch GPU trên Windows
│   └── readme_linux.md               # Hướng dẫn chi tiết biên dịch GPU trên Linux
│
├── data/                             # Thư mục lưu trữ tập dữ liệu (CSV)
│   └── real/
│       ├── iris/                     # Dataset Iris (150 mẫu, 3 cụm)
│       └── fashion-mnist/            # Dataset Fashion-MNIST (70,000 mẫu, 10 cụm)
│
├── theory/
│   └── THEORY.md                     # Tài liệu toán học & ánh xạ công thức của thuật toán
│
├── visual/                           # Thư mục chứa script phân tích cấu trúc dữ liệu gốc
│   └── visualize_data.py
│
├── dpc_aknn_optimization_presentation.html # Báo cáo chi tiết tối ưu hóa & song song hóa
└── README.md                         # Tài liệu hướng dẫn này
```

---

## 💻 Yêu Cầu Môi Trường (Prerequisites)

### 1. Biên dịch C/C++ và CUDA
* **Hệ điều hành:** Windows (khuyên dùng Windows 10/11) hoặc Linux (Ubuntu 20.04 trở lên).
* **CPU compiler:** `GCC` (hỗ trợ OpenMP). Trên Windows sử dụng bộ **MinGW-w64**.
* **GPU compiler (cho GPU parallel):** `NVCC` (CUDA Toolkit 11.x / 12.x trở lên).
* **MSVC Compiler (Windows):** `cl.exe` đi kèm Visual Studio 2022 (được NVCC gọi để build host-code).
* **Build Tool:** `make` (Linux) hoặc `mingw32-make` (Windows).

### 2. Môi trường Python (Dùng cho Demo & Đánh giá)
Cần cài đặt Python 3.8+ và các thư viện cần thiết:
```bash
pip install numpy matplotlib scikit-learn scipy pandas
```

---

## 🛠️ Hướng Dẫn Biên Dịch & Chạy Chương Trình

### 1. Phiên Bản CPU (OpenMP)

Bản CPU được cấu hình mặc định để phân cụm tập dữ liệu **Iris** (3 cụm, $K = 7$).

#### 🔹 Biên dịch:
* **Trên Windows (PowerShell/CMD có MinGW):**
  ```powershell
  cd cpu_parallel
  mingw32-make clean
  mingw32-make
  ```
* **Trên Linux (Terminal):**
  ```bash
  cd cpu_parallel
  make clean
  make
  ```

#### 🔹 Chạy chương trình:
* **Cách 1: Chạy bằng cấu hình mặc định (Iris dataset):**
  ```bash
  # Windows
  .\dpc_aknn_cpu.exe
  # Linux
  ./dpc_aknn_cpu
  ```
* **Cách 2: Truyền tham số dòng lệnh để chạy dataset khác (ví dụ: Fashion-MNIST):**
  ```bash
  ./dpc_aknn_cpu --input ../data/real/fashion-mnist/fashion_mnist_X.csv --labels ../data/real/fashion-mnist/fashion_mnist_y.csv --clusters 10 --k 15
  ```

---

### 2. Phiên Bản GPU (CUDA)

Bản GPU được cấu hình mặc định để phân cụm tập dữ liệu lớn **Fashion-MNIST** (10 cụm, $K = 15$).

#### 🔹 Biên dịch:
* **Trên Windows:**
  Mở **Developer PowerShell for VS 2022** (để tự động nạp `cl.exe` vào PATH) và chạy:
  ```powershell
  cd gpu_parallel
  mingw32-make clean
  mingw32-make
  ```
  *(Lưu ý: Nếu sử dụng PowerShell thường, bạn cần gán thủ công đường dẫn chứa `cl.exe` vào PATH trước khi build, ví dụ: `$env:PATH += ";C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\<Version>\bin\Hostx64\x64"`)*
  
* **Trên Linux:**
  ```bash
  cd gpu_parallel
  make clean
  make
  ```

#### 🔹 Chạy chương trình:
* **Cách 1: Chạy bằng cấu hình mặc định (Fashion-MNIST):**
  ```bash
  # Windows
  .\dpc_aknn_gpu.exe
  # Linux
  ./dpc_aknn_gpu
  ```
* **Cách 2: Truyền tham số dòng lệnh:**
  ```bash
  ./dpc_aknn_gpu --input ../data/real/iris/iris_X_norm.csv --labels ../data/real/iris/iris_y.csv --clusters 3 --k 7
  ```

---

## ⚙️ Cấu Hình Tham Số Mặc Định (config.h)

Bạn có thể chỉnh sửa các tham số chạy mặc định trực tiếp trong file `config.h` của từng phiên bản:

### 📍 CPU Configuration ([cpu_parallel/src/config.h](file:///D:/BTL_LTSS/cpu_parallel/src/config.h))
* `DEFAULT_INPUT_FILE` & `DEFAULT_LABEL_FILE`: Đường dẫn dữ liệu và nhãn kiểm tra.
* `DEFAULT_N_CLUSTERS`: Số lượng cụm cần phân chia.
* `DEFAULT_K`: Số láng giềng gần nhất cho mỗi điểm.
* `OMP_N_THREADS`: Thiết lập số luồng OpenMP (đặt `0` để sử dụng toàn bộ lõi CPU khả dụng).

### 📍 GPU Configuration ([gpu_parallel/src/config.h](file:///D:/BTL_LTSS/gpu_parallel/src/config.h))
* Ngoài các tham số cơ bản tương tự CPU, bản GPU hỗ trợ thêm cấu hình phần cứng:
  * `GPU_DEVICE_ID`: Chọn GPU để thực thi (mặc định: `0`).
  * `GPU_BATCH_SIZE`: Kích thước phân đoạn tính toán trên GPU nhằm tối ưu hóa dung lượng VRAM (mặc định: `1000`).
  * `BLOCK_SIZE_1D` & `BLOCK_SIZE_2D_X/Y`: Cấu hình số lượng Thread Block cho các CUDA kernel 1 chiều và 2 chiều.

---

## 📊 Trực Quan Hóa & Đánh Giá Kết Quả

Sau khi chạy phiên bản C hoặc CUDA, nhãn phân cụm dự đoán sẽ được lưu tại thư mục:
* CPU: `cpu_parallel/output/labels/cpu_labels_<timestamp>.csv`
* GPU: `gpu_parallel/output/labels/gpu_labels_<timestamp>.csv`

### 1. Xem Chỉ Số Đánh Giá Chất Lượng (ARI, NMI, ACC)
Bạn có thể sử dụng công cụ `metrics.py` của dự án để đánh giá độ chính xác phân cụm bằng **thuật toán Hungarian** (khớp tối ưu hóa nhãn dự đoán và nhãn thực tế):
```bash
python gpu_parallel/utility_functions/metrics.py --pred gpu_parallel/output/labels/<tên_file_output>.csv --true data/real/fashion-mnist/fashion_mnist_y.csv
```

### 2. Vẽ Biểu Đồ So Sánh Ground-Truth vs Dự Đoán
Sử dụng script `visualize.py` để chiếu giảm chiều không gian đặc trưng bằng PCA hoặc t-SNE và biểu diễn các cụm dưới dạng màu sắc:
```bash
# Trực quan hóa kết quả phân cụm trên tập Iris
python cpu_parallel/utility_functions/visualize.py --pred cpu_parallel/output/labels/<tên_file_output>.csv --true data/real/iris/iris_y.csv --data data/real/iris/iris_X_norm.csv --clusters 3 --method pca
```

### 3. Chạy Bản Python Live-Demo (Cập nhật đồ họa thời gian thực)
Dự án có cung cấp một bản demo viết bằng Python, thực thi từng bước trong 8 bước của thuật toán DPC-AKNN trên giao diện đồ họa động:
```bash
python gpu_parallel/utility_functions/demo.py --data data/real/fashion-mnist/fashion_mnist_X.csv --labels data/real/fashion-mnist/fashion_mnist_y.csv --clusters 10 --k 15 --samples 5000
```

---

## 📖 Cơ Sở Toán Học & Ánh Xạ Mã Nguồn
Để tìm hiểu sâu hơn về cơ sở lý thuyết, các công thức toán học (Equation 1 - 12) trong bài báo và cách chúng được cài đặt bằng mã nguồn C/CUDA, vui lòng xem tài liệu chi tiết tại:
👉 [theory/THEORY.md](file:///D:/BTL_LTSS/theory/THEORY.md)
