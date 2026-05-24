import pandas as pd
import os

# Đường dẫn
input_path = r'C:\BTL_LTSS\data\real\iris\Iris.csv'
output_dir = r'C:\BTL_LTSS\data\real\iris'

# Đọc dữ liệu
df = pd.read_csv(input_path)

# 1. Tách đặc trưng (bỏ Id và Species)
# Iris có 4 đặc trưng: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
# 1. Tách 4 cột đặc trưng (bỏ Id và Species)
X = df.iloc[:, 1:5]

# 2. Chuẩn hóa Min-Max về [0, 1]
# Lý do: Min-Max cho ACC=0.9667 (k=7), tốt hơn Z-score (ACC=0.8667)
X_norm = (X - X.min()) / (X.max() - X.min())
X_norm.to_csv(os.path.join(output_dir, 'iris_X_norm.csv'), index=False, header=False)

# 3. Chuyển nhãn chuỗi sang số
# Iris-setosa -> 0, Iris-versicolor -> 1, Iris-virginica -> 2
y = df['Species'].astype('category').cat.codes
y.to_csv(os.path.join(output_dir, 'iris_y.csv'), index=False, header=False)

print(f"Created iris_X.csv ({X.shape}) and iris_y.csv ({y.shape}) at {output_dir}")
