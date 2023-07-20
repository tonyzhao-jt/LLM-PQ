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
    # remove columns input_seq_length != 384 and 768
    df = df[df['input_seq_length'].isin([384, 768])]
    # restore the new csv file
    df.to_csv(f, index=False)