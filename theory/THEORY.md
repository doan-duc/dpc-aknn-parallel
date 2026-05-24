# Cơ sở lý thuyết DPC-AKNN

## 1. Động cơ

DPC-AKNN cải tiến DPC gốc ở hai điểm chính:

- Hàm mật độ không còn dùng toàn bộ tập dữ liệu hay ngưỡng cắt tĩnh, mà ràng buộc theo lân cận gần nhất.
- Giai đoạn gán cụm không thực hiện kiểu truyền nhãn một bước dễ lan lỗi, mà dùng ma trận liên kết, cập nhật lặp, rồi hiệu chỉnh bằng bỏ phiếu kNN.

## 2. Hàm mật độ cục bộ

Với mỗi điểm `x_i`, mật độ cục bộ được tính theo:

`rho_i = sum_{x_j in N^k_{x_i}} exp(-(d_ij^2) / (d_c^2))`

Ý nghĩa:

- Chỉ dùng `k` láng giềng gần nhất nên giảm nhiễu từ điểm xa.
- Vẫn giữ dạng Gaussian nên tận dụng được thông tin khoảng cách, tốt hơn cutoff kernel.

## 3. Ngưỡng toàn cục `d_c`

Đầu tiên tính ngưỡng cục bộ từng điểm:

`d_ci = mean(knn_dist_i) + std(knn_dist_i)`  theo Eq. (6)

Sau đó suy ra:

- `d_bar_c = mean(d_ci)` theo Eq. (8)
- `sigma_c = std(d_ci, ddof=1)` theo Eq. (9)
- `d_c = d_bar_c + sigma_c` theo Eq. (7)

Thiết kế này giúp `d_c` thích nghi với phân bố cục bộ nhưng vẫn là ngưỡng toàn cục dùng thống nhất cho toàn bộ thuật toán.

## 4. Chọn tâm cụm

Khoảng cách tới điểm có mật độ cao hơn được tính theo:

`delta_i = min_{j: rho_j > rho_i} d_ij`

Riêng điểm có mật độ lớn nhất thì:

`delta_i = max_j d_ij`

Sau đó dùng:

`gamma_i = rho_i * delta_i`

và chọn `n_c` điểm có `gamma` lớn nhất làm tâm cụm.

## 5. Xây cụm ban đầu

Mỗi tâm cụm `c_j` khởi tạo một cụm `C_j` gồm:

- chính tâm `c_j`
- toàn bộ `k` láng giềng của tâm

Tiếp theo cụm được mở rộng theo luật admission. Với `x_p` đã nằm trong cụm, xét `x_q` thuộc kNN của `x_p`, chỉ thêm `x_q` nếu:

1. `x_q` chưa được gán nhãn
2. `d_pq <= mean distance từ x_p đến kNN của nó`
3. `distance(centroid_j, x_q) <= d_c`

Điều kiện 2 là trung bình, không phải cực đại. Điều kiện 3 dùng `d_c` toàn cục.

## 6. Gán điểm mật độ thấp bằng ma trận liên kết

Với tập điểm chưa gán `U`, dựng ma trận:

`A(i, r) = sum_{x_l in N^k_{x_i} intersect C_r} (1 / d_il) * rho_l * rho_i`

Sau đó lặp:

1. Chọn phần tử lớn nhất `(i*, r*)`
2. Gán `x_{i*}` vào cụm `r*`
3. Xóa hàng `i*`
4. Chỉ cập nhật cột `r*` theo Eq. (11) cho các điểm chưa gán có `x_{i*}` trong kNN của chúng

Đây là chỗ có độ phức tạp chi phối toàn thuật toán.

## 7. Hiệu chỉnh lỗi

Sau khi mọi điểm đã được gán sơ bộ:

- Sắp các điểm theo `rho` giảm dần
- Với mỗi điểm, đếm phiếu nhãn trong `k` láng giềng
- Gán về cụm có số phiếu lớn nhất
- Nếu hòa, chọn cụm có khoảng cách trung bình nhỏ nhất tới các láng giềng thuộc cụm đó

## 8. Gán nốt điểm còn lại

Nếu vẫn còn điểm chưa có nhãn, với mỗi cụm `r` tính:

`zeta_ir = mean distance từ x_i đến k láng giềng gần nhất nằm trong cụm r`

và chọn:

`r* = argmin_r zeta_ir`

## 9. Độ phức tạp

Theo bài báo, thành phần chi phối là vòng lặp cập nhật ma trận liên kết, dẫn đến tổng thể:

`O(n_c * n^2)`

Đây là lý do cần thêm hai bản song song hóa:

- `cpu_parallel/` dùng OpenMP
- `gpu_parallel/` dùng CUDA
