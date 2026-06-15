# DPC-AKNN CPU (Windows)

### 1. Biên dịch (Compile)
Mở PowerShell (hoặc Command Prompt có hỗ trợ `gcc`) tại thư mục này và chạy:
```powershell
# Dọn dẹp các file build cũ
mingw32-make clean

# Biên dịch chương trình
mingw32-make
```
### 2. Chạy chương trình (Run)

```powershell
.\dpc_aknn_cpu.exe
```
