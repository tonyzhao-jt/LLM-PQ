import pandas as pd

input_file = "Tesla_T4_13b.csv"
df = pd.read_csv(input_file)
df = df[df["bit"] != "8:tc"]
df.to_csv(input_file, index=False)