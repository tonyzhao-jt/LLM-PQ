# remove entry with column with specified value
# read all files under the folder
import os

files = os.listdir(".")
# get csv files
csv_files = [f for f in files if f.endswith(".csv")]
# load with pandas
import pandas as pd
for f in csv_files:
    df = pd.read_csv(f)
    # remove columns input_seq_length == '8:tc-li'
    # df = df[~df['bit'].isin(['8:tc-li'])]
    # restore the new csv file
    file_name_front = f.split(".csv")[0]
    new_file_name = f"{file_name_front}_decode.csv"
    df.to_csv(new_file_name, index=False)