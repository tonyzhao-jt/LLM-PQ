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
    file_name = f.split('.csv')
    file_name_front = file_name[0]
    # remove columns input_seq_length != '8:tc-li'
    # df = df[df['past_seq_length'].isin([0])] # remove all decode stage
    # new_name = f"{file_name_front}_prefill.csv"
    # restore the new csv file
    # change name to *_prefill.csv

    # df = df[~df['bit'].isin(['8:tc-li'])] # remove all 8bit
    # new_name = f
    df = df[df['bit'].isin(['8:tc-li'])] # get all 8bit
    new_name = f"{file_name_front}_prefill_8_bit.csv"

    df.to_csv(new_name, index=False)