# Bước 1: C binary chạy toàn bộ 70K
cd C:\BTL_LTSS\cpu_parallel
.\dpc_aknn_cpu.exe --input ../data/real/fashion-mnist/fashion_mnist_X.csv --labels ../data/real/fashion-mnist/fashion_mnist_y.csv --clusters 10 --k 15

# Bước 2: Python vẽ kết quả
cd utility_functions
python visualize.py --pred C:\BTL_LTSS\cpu_parallel\output\labels\cpu_labels_20260511_114100.csv --true ../../data/real/fashion-mnist/fashion_mnist_y.csv --data ../../data/real/fashion-mnist/fashion_mnist_X.csv

# Chạy dataset spiral
.\dpc_aknn_cpu.exe --input ../data/artificial/spiral_X.csv --labels ../data/artificial/spiral_y.csv --clusters 2 --k 15

cd utility_functions
python visualize.py --pred ../output/labels/cpu_labels_20260511_114100.csv --true ../../data/artificial/spiral_y.csv --data ../../data/artificial/spiral_X.csv --clusters 2


.\dpc_aknn_cpu.exe --input ../data/artificial/3-spiral_X.csv --labels ../data/artificial/3-spiral_y.csv --clusters 3 --k 15
