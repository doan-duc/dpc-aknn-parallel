# DPC-AKNN Project

Dự án này triển khai thuật toán DPC-AKNN theo bài báo:

> Houshen Lin, Jian Hou, Huaqiang Yuan. "Density peak clustering based on nearest neighbors". Engineering Applications of Artificial Intelligence 160 (2025) 111981.

## Cấu trúc

- `data/`: dữ liệu dùng chung cho cả 3 phiên bản
- `theory/`: tóm tắt lý thuyết và ánh xạ công thức
- `original/`: bản tham chiếu Python + NumPy
- `cpu_parallel/`: bản C99 + OpenMP
- `gpu_parallel/`: bản CUDA C

## Lưu ý

- `data/real/` hiện đang để trống.
- Demo va benchmark yeu cau dataset CSV trong `data/real/`.
- Bản `cpu_parallel/` và `gpu_parallel/` đã được scaffold theo đúng giao diện prompt, nhưng cần môi trường biên dịch phù hợp để xác minh thực thi.

## Trạng thái xác minh

- `original/`: kiểm tra trực tiếp `fit_predict()` đã chạy được; `pytest` chưa có trong môi trường hiện tại.
- `cpu_parallel/`: đã compile bằng GCC trực tiếp, test PASS và demo ghi output thành công.
- `gpu_parallel/`: đã thêm source CUDA; `nvcc` hiện bị chặn vì thiếu `cl.exe` trong PATH trên Windows.
