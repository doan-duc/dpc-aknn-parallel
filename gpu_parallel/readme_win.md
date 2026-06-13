# DPC-AKNN GPU (Windows)

### 1. Biên dịch (Compile)
Mở **Developer PowerShell for VS 2022** tại thư mục này và chạy:
```powershell
# Dọn dẹp các file build cũ
mingw32-make clean

# Biên dịch chương trình
mingw32-make
```

*(Lưu ý: Nếu dùng PowerShell thường, cần nạp đường dẫn chứa `cl.exe` của MSVC trước, ví dụ: `$env:PATH += ";C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64"` rồi mới chạy hai lệnh trên).*

### 2. Chạy chương trình (Run)

**Cách 1: Chạy bằng cấu hình mặc định (được thiết lập trong file `config.h`)**
```powershell
.\dpc_aknn_gpu.exe
```



