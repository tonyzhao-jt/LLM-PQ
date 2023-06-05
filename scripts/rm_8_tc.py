import os
import pandas as pd
import argparse

def remove_bit_rows(folder):
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            file_path = os.path.join(folder, file)
            df = pd.read_csv(file_path)
            if "bit" in df.columns:
                df = df[df["bit"] != "8:tc-li"]
            df.to_csv(file_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove rows with bit == 8:tc-li from CSV files in a folder.")
    parser.add_argument("--folder", type=str, default="./lat_profiled_result", help="Path to the folder containing the CSV files.")
    args = parser.parse_args()

    remove_bit_rows(args.folder)