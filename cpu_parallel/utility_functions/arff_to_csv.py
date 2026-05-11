import argparse
import os
from scipy.io import arff
import pandas as pd
import numpy as np

def convert_arff(input_arff, output_dir=None):
    print(f"[ARFF] Dang doc {input_arff}...")
    
    # Đọc dữ liệu arff
    data, meta = arff.loadarff(input_arff)
    df = pd.DataFrame(data)
    
    # Cột cuối cùng thường là nhãn (y)
    label_col = df.columns[-1]
    
    X = df.drop(columns=[label_col]).values
    y_raw = df[label_col].values
    
    # Nếu nhãn là chuỗi bytes (do arff parse ra bytes), decode nó
    if isinstance(y_raw[0], bytes):
        y_raw = [val.decode('utf-8') for val in y_raw]
    
    # Chuyển nhãn thành số nguyên (0, 1, 2, ...)
    unique_labels = sorted(list(set(y_raw)))
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    y = np.array([label_map[lbl] for lbl in y_raw], dtype=np.int32)
    
    print(f"[ARFF] Da tim thay {len(unique_labels)} class: {label_map}")
    print(f"[ARFF] X shape: {X.shape}, y shape: {y.shape}")
    
    if output_dir is None:
        output_dir = os.path.dirname(input_arff)
        if not output_dir: output_dir = "."
        
    base_name = os.path.splitext(os.path.basename(input_arff))[0]
    out_X = os.path.join(output_dir, f"{base_name}_X.csv")
    out_y = os.path.join(output_dir, f"{base_name}_y.csv")
    
    print(f"[ARFF] Dang luu X vao: {out_X}...")
    np.savetxt(out_X, X, delimiter=',', fmt='%g')
    
    print(f"[ARFF] Dang luu y vao: {out_y}...")
    np.savetxt(out_y, y, delimiter=',', fmt='%d')
    
    print("[ARFF] Xong! Co the dung luon 2 file nay cho C binary.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuyen doi file ARFF sang X.csv va y.csv")
    parser.add_argument("input", help="Duong dan den file .arff")
    parser.add_argument("--outdir", help="Thu muc luu (mac dinh: cung cho voi file goc)", default=None)
    args = parser.parse_args()
    
    convert_arff(args.input, args.outdir)
