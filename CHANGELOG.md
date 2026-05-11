# CHANGELOG

## Phiên bản khởi tạo 2026-05-09

- Triển khai cấu trúc dự án theo prompt gốc cho `original/`, `cpu_parallel/`, `gpu_parallel/`, `theory/`, `data/`.
- Viết bản tham chiếu Python/NumPy với đầy đủ các bước Eq. (4) đến Eq. (12).
- Viết bản C99 + OpenMP với cùng luồng thuật toán và giao diện dòng lệnh.
- Viết bản CUDA C với kernel cho ma trận khoảng cách, mật độ `rho`, ma trận liên kết `A` và bỏ phiếu kNN; các bước tuần tự còn lại chạy ở host.
- Demo va benchmark hien yeu cau file CSV do nguoi dung cung cap.
- Chưa xác minh thực thi thực tế của `cpu_parallel/` và `gpu_parallel/` trên máy hiện tại vì môi trường phiên này không build/chạy được trình biên dịch C/OpenMP/CUDA.
- Chưa đối chiếu ARI = 1.0 giữa ba phiên bản vì chưa có pipeline benchmark liên ngôn ngữ đang chạy thật trong phiên này.

## Cập nhật 2026-05-09

- Bổ sung đầy đủ source cho `cpu_parallel/`: `config.h`, `csv_io.*`, `utils_omp.*`, `dpc_aknn.*`, `main.c`, `test_dpc_aknn.c`, `Makefile`, `README.md`.
- Bản CPU đã được biên dịch trực tiếp bằng GCC vì máy hiện tại không có `make` trong PATH.
- Test CPU đã PASS toàn bộ bằng lệnh GCC trực tiếp tương đương target `make test`.
- Demo CPU đã chạy thành công và ghi nhãn vào `cpu_parallel/outputs/labels/`.
- Bổ sung source CUDA cho `gpu_parallel/`: `config.h`, `kernels.cu/.cuh`, `utils_gpu.*`, `dpc_aknn.*`, `main.cu`, `test_dpc_aknn.cu`, `Makefile`, `README.md`.
- Chưa xác minh build CUDA vì `nvcc` báo thiếu MSVC `cl.exe` trong PATH trên Windows: `nvcc fatal : Cannot find compiler 'cl.exe' in PATH`.
- B?n Python `original/` ch?a ch?y `pytest` v? m?i tr??ng thi?u module `pytest`.
