import os
import pandas as pd

# specify the path to the folder containing the csv files
folder_path = "."

# get a list of all csv files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]

# iterate over each csv file
for csv_file in csv_files:
    # read the csv file into a pandas dataframe
    df = pd.read_csv(os.path.join(folder_path, csv_file))
    # remove the rows where "bit" is equal to "8:tc" or "8:tc-li"
    df = df[~df["bit"].isin(["8:tc", "8:tc-li"])]
    # overwrite the original csv file with the updated dataframe
    df.to_csv(os.path.join(folder_path, csv_file), index=False)