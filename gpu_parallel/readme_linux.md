# DPC-AKNN GPU (Linux)

Phiên bản CUDA C của thuật toán DPC-AKNN được tối ưu hóa lai giữa GPU và CPU (Hybrid Parallelization) trên môi trường Linux.

## Hướng dẫn Biên dịch (Compile)
Đảm bảo bạn đã cài đặt `nvcc` (CUDA Toolkit) và `make`. Bạn có thể kiểm tra xem hệ thống đã cài đặt chưa bằng lệnh `nvcc --version`.
Mở terminal tại thư mục mã nguồn và chạy lệnh sau để dọn dẹp và biên dịch:

```bash
make clean
make
```
Lệnh này sẽ tự động gọi `nvcc` kèm theo thư viện `-lcublas` để sinh ra file thực thi `dpc_aknn_gpu`.

## Hướng dẫn Chạy Test (Run)
Chương trình được thiết kế chạy với **đường dẫn tương đối** (dấu `/` trên Linux), giúp dễ dàng clone và chạy trên bất kỳ server hoặc máy tính nào mà không cần sửa code.

Ví dụ chạy test với tập dữ liệu Fashion-MNIST:
```bash
./dpc_aknn_gpu --input ../data/real/fashion-mnist/fashion_mnist_X.csv --labels ../data/real/fashion-mnist/fashion_mnist_y.csv --clusters 10 --k 15
```

## Kết quả và Log (Logging & Profiling)
Quá trình chạy trên Linux sẽ xuất log rất chi tiết ra terminal:
- **Theo dõi thời gian từng bước:** Hiển thị thời gian cụ thể (tính bằng giây) cho từng công đoạn như kNN, tính d_c, rho, delta, BFS, cập nhật ma trận liên kết A và Voting. Điều này rất hữu ích cho quá trình tối ưu hóa.
- **Tổng kết hiệu năng & chất lượng:** Cuối cùng, chương trình sẽ in ra một bảng tổng kết chứa **tổng thời gian** xử lý và các chỉ số so khớp với nhãn thật: **ARI**, **NMI**, **ACC**.
- File nhãn kết quả (`.csv`) và log thực thi (`.txt`) được tự động lưu lại vào thư mục `output/labels/` và `output/logs/`.
