# create the latency cost model w.r.t the h1, h2, b and i 
b = 16  # Set the value of b
h1 = 8  # Set the value of h1
h2 = 4  # Set the value of h2

def PROJ_MOPS(b, h1):
    return 2 * b * h1 + h1 ** 2

def BMM_MOPS(b, h1, i):
    return b * h1 + b * h1 * i + b * i 

def MLP_MOPS(b, h1, h2, i):
    return b * h1 + b * h2 + h1 * h2

def SELF_ATTN_MOPS_PARMAS(b, h1, i, bit):
    weight_size = 4 * h1 * h1 * 2 # fp16 bydefault
    qkv_act_size = 2 * 4 * b * 1 * h1 * 2 # fp16 bydefault
    bmm_act_size = 4 * b * h1 * i * 2 + 2 * b * i # fp16 bydefault
    kv_concat_size = 2 * b * i * h1 * 2 # fp16 bydefault
    layer_norm_size = b * i * h1 * 2 # fp16 bydefault
    dequant_size = 0
    bit = int(bit) if str(bit).isnumeric() else bit
    if bit == '8:tc':
        # it reduce both mops for weight and activation
        # but also reduce the concat/split cost for KV (int8)
        weight_size = weight_size / 2
        qkv_act_size = qkv_act_size / 2
        kv_concat_size = kv_concat_size / 2
        bmm_act_size = bmm_act_size / 2
        layer_norm_size = layer_norm_size / 2
    else:
        if bit == '8:tc-li':
            weight_size = weight_size / 2
            dequant_size = qkv_act_size
        elif bit == 8:
            weight_size = weight_size / 2
            dequant_size = weight_size
        elif bit == 4:
            weight_size = weight_size / 4
            dequant_size = weight_size
        elif bit == 2:
            weight_size = weight_size / 8
            dequant_size = weight_size

    return weight_size, qkv_act_size, kv_concat_size, bmm_act_size, layer_norm_size, dequant_size

def FFN_MOPS_PARAMS(b, h1, h2, bit):
    weight_size = 2 * h1 * h2 * 2 # fp16 bydefault
    act_size = b * (h1 + h2) * 2 # fp16 bydefault
    layer_norm_size = b * h1 * 2
    bit = int(bit) if str(bit).isnumeric() else bit
    if bit == '8:tc':
        # it reduce both mops for weight and activation
        weight_size = weight_size / 2
        act_size = act_size / 2
        layer_norm_size = layer_norm_size / 2
        dequant_size = 0
    else:
        if bit == '8:tc-li':
            weight_size = weight_size / 2
            dequant_size = act_size
        elif bit == 8:
            weight_size = weight_size / 2
            dequant_size = weight_size
        elif bit == 4:
            weight_size = weight_size / 4
            dequant_size = weight_size
        elif bit == 2:
            weight_size = weight_size / 8
            dequant_size = weight_size

    return weight_size, act_size, layer_norm_size, dequant_size

import os 
import pandas as pd 

# read the profiled result from folder
profiled_result_folder = '.'
target_device = 'V100'
# list the dir to find the deviced related profile result
profiled_result_files = os.listdir(profiled_result_folder)
# filter the file with the target device
profiled_result_files = [os.path.join(profiled_result_folder, f) for f in profiled_result_files if target_device in f]
# read all the profiled result and form a big df
df = pd.concat([pd.read_csv(f) for f in profiled_result_files])
# average result with same column value for : shard,h1,h2,bit,batch_size,input_seq_length,past_seq_length,
df = df.groupby(['shard','h1','h2','bit','batch_size','input_seq_length','past_seq_length']).mean().reset_index()
profile_df = df 

# first get shard = 0: ATTENTION MODELS
# then get shard = 1: FFN MODELS
shard_0_profiled_result = profile_df[profile_df['shard'] == 0]
df[["weight_size", "qkv_act_size", "kv_concat_size", "bmm_act_size", "layer_norm_size", "dequant_size"]] = df.apply(
    lambda row: SELF_ATTN_MOPS_PARMAS(row['batch_size'], row['h1'], row['input_seq_length'] + row['past_seq_length'], row['bit']) \
    , axis=1, result_type='expand')

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import numpy as np 
X = df[["weight_size", "qkv_act_size", "kv_concat_size", "bmm_act_size", "layer_norm_size", "dequant_size"]]
X = sm.add_constant(X)
y = df["lat_avg"]
# model = sm.OLS(y, X).fit()
# # Print the model summary
# print(model.summary())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit an OLS regression model on the training data
X_train = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train).fit()

# Print the model summary
print(model.summary())

# Generate predictions for the testing data
X_test = sm.add_constant(X_test)
y_pred = model.predict(X_test)

# Compute the prediction errors (residuals) for the testing data
residuals = y_test - y_pred

# Print the mean squared error (MSE) and the mean absolute error (MAE) for the testing data
mse = np.mean(residuals**2)
mae = np.mean(abs(residuals))
print("Mean squared error (MSE) for the testing data:", mse)
print("Mean absolute error (MAE) for the testing data:", mae)

import pickle
with open(f"{target_device}_lat_model.pkl", "wb") as f:
    pickle.dump(model, f)