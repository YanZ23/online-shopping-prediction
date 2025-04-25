
"""
Data-pre-processing script.

Steps
-----
1. Read the raw CSV file.
2. One-Hot-Encode all categorical columns.
3. Standard-scale all numeric columns (except target).
4. Save the processed dataset to disk.

CLI
---
python src/preprocess.py \
  --input  data/online_shoppers_intention.csv \
  --output data/preprocessed_data.csv
"""

import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess(input_path: str, output_path: str) -> None:
    # 1. Load
    df = pd.read_csv(input_path)
    print(f"[INFO] Raw shape: {df.shape}")

    # 2. One-Hot-Encode object columns
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print(f"[INFO] After encoding: {df.shape}")

    # 3. Scale numeric columns except target
    num_cols = df.drop("Revenue", axis=1).select_dtypes(include="number").columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # 4. Save
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Pre-processed dataset saved → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-process Online Shoppers Intention data")
    parser.add_argument("--input", "-i", default="../data/online_shoppers_intention.csv",
                        help="Path to the raw CSV file")
    parser.add_argument("--output", "-o", default="../data/preprocessed_data.csv",
                        help="Path to save the processed CSV")
    args = parser.parse_args()
    preprocess(args.input, args.output)


if __name__ == "__main__":
    main()