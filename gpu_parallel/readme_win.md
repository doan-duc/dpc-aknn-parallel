# DPC-AKNN GPU (Windows)

Phiên bản CUDA C của thuật toán DPC-AKNN được tối ưu hóa lai giữa GPU và CPU (Hybrid Parallelization) trên môi trường Windows.

## Hướng dẫn Biên dịch (Compile)
Đảm bảo bạn đã cài đặt CUDA Toolkit và MSVC (Microsoft Visual Studio C++ Compiler).
Mở PowerShell, khai báo biến môi trường cho MSVC (đường dẫn có thể khác tùy máy), sau đó sử dụng `mingw32-make` để biên dịch:

```powershell
# Nạp biến môi trường MSVC (Sửa đường dẫn nếu cần)
$env:PATH += ";C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"

# Dọn dẹp và biên dịch
mingw32-make clean
mingw32-make
```

## Hướng dẫn Chạy Test (Run)
Chương trình sử dụng **đường dẫn tương đối** (relative paths) giúp bạn có thể chạy trên bất kỳ máy nào chứa project mà không bị lỗi đường dẫn cứng.

Ví dụ chạy test với tập dữ liệu Fashion-MNIST trên Windows:
```powershell
.\dpc_aknn_gpu.exe --input ..\data\real\fashion-mnist\fashion_mnist_X.csv --labels ..\data\real\fashion-mnist\fashion_mnist_y.csv --clusters 10 --k 15
```

## Kết quả và Log (Logging & Profiling)
Trong quá trình chạy, màn hình console sẽ hiển thị **thời gian thực thi cực kì chi tiết theo từng bước** (kNN, d_c, rho, delta, BFS, Ma trận liên kết A, Voting...).
Kết thúc quá trình, chương trình in ra bảng tổng kết với:
- **Tổng thời gian chạy** toàn bộ pipeline thuật toán.
- **Các chỉ số độ chính xác:** ARI (Adjusted Rand Index), NMI (Normalized Mutual Info), ACC (Clustering Accuracy).
- Toàn bộ nội dung console và nhãn dự đoán sẽ được tự động lưu ra file vào các thư mục `output/logs/` và `output/labels/`.
