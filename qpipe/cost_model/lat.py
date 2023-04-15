import os 
import numpy as np 
import pickle
import pandas as pd 
import statsmodels.api as sm

from . import mops 

def fit_cost_model(profiled_result_folder, cost_model_store_path, target_device=None):
    from sklearn.model_selection import train_test_split

    assert os.path.exists(profiled_result_folder), f"Folder {profiled_result_folder} does not exist."
    assert target_device, "target_device cannot be None"
    # read the profiled result from folder
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
    for target_shard in [0, 1]:
        df = profile_df[profile_df['shard'] == target_shard]
        df = df[df['lat_avg'] < 99998]

        if target_shard == 0:
            df[["weight_size", "qkv_act_size", "kv_concat_size", "bmm_act_size", "layer_norm_size", "dequant_size"]] = df.apply(
                lambda row: mops.SELF_ATTN_MOPS_PARAMS(row['batch_size'], row['h1'], row['input_seq_length'] + row['past_seq_length'], row['bit']) \
                , axis=1, result_type='expand')
            X = df[["weight_size", "qkv_act_size", "kv_concat_size", "bmm_act_size", "layer_norm_size", "dequant_size"]]
        else:
            df[["weight_size", "act_size", "layer_norm_size", "dequant_size"]] = df.apply(
                lambda row: mops.FFN_MOPS_PARAMS(row['batch_size'], row['h1'], row['h2'], row['bit']) \
                , axis=1, result_type='expand')
            X = df[["weight_size", "act_size", "layer_norm_size", "dequant_size"]]
        X = sm.add_constant(X)
        y = df["lat_avg"]

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

        with open(f"{cost_model_store_path}/{target_device}_{target_shard}_lat_model.pkl", "wb") as f:
            pickle.dump(model, f)

def find_pairs(n):
    pairs = []
    for i in range(1, n+1):
        if n % i == 0:
            pairs.append((i, n//i))
    return pairs

class LatCostModel:
    def __init__(self, lat_cost_model_folder, device_names) -> None:
        assert os.path.exists(lat_cost_model_folder), f"Folder {lat_cost_model_folder} does not exist."
        self.device_names = device_names
        self.cost_model = {}
        # for each device, load the cost model
        for device_name in device_names:
            self.cost_model[device_name] = {}
            self_attn_cost_model_name = f"{device_name}_{0}"
            ffn_cost_model_name = f"{device_name}_{1}"
            for file in os.listdir(lat_cost_model_folder):
                file_path = os.path.join(lat_cost_model_folder, file)
                if 'lat_model' in file and file.endswith('.pkl') and device_name in file:
                    if self_attn_cost_model_name in file:
                        with open(file_path, 'rb') as f:
                            self.cost_model[device_name][0] = pickle.load(f)
                    elif ffn_cost_model_name in file:
                        with open(file_path, 'rb') as f:
                            self.cost_model[device_name][1] = pickle.load(f)
        # make sure that each device has two cost model
        for device_name in device_names:
            assert len(self.cost_model[device_name]) == 2, f"Cannot find cost model for {device_name}"

        self.has_hypers = False
        self.has_profiled = False
        self.profiled_data = {}

    def device_in_cost_model(self, device_name):
        return device_name in self.device_names
    
    def register_hyper_params(self, b, i, h1, h2):
        self.b = b
        self.i = i
        self.h1 = h1
        self.h2 = h2
        self.has_hypers = True

    def get_available_chunks(self):
        # get all prime factorization of b
        assert self.has_hypers, "Hyper params not registered."
        available_paris = find_pairs(self.b) # return micro batch size and number of chunks
        return available_paris
    
    def change_bs(self, b):
        assert self.has_hypers, "Hyper params not registered."
        self.b = b

    
    def fetch_lat_with_hyper(self, shard, device_name, bit):
        profiled_data_device = self.profiled_data[device_name]
        # columns: shard,h1,h2,bit,batch_size,input_seq_length,past_seq_length,lat_avg,mem_weight,mem_kv,mem_embedding,mem_all
        # fetch data with hyper params
        profiled_data_device = profiled_data_device[(profiled_data_device['shard'] == shard) & 
                                            (profiled_data_device['h1'] == self.h1) &
                                            (profiled_data_device['h2'] == self.h2) &
                                            (profiled_data_device['bit'] == str(bit)) &
                                            (profiled_data_device['batch_size'] == self.b) &
                                            (profiled_data_device['past_seq_length'] == self.i)]
        if len(profiled_data_device) == 0:
            return None
        # lat_avg
        lat_avg = profiled_data_device['lat_avg'].values[0]
        return lat_avg

    def update_profiled_result(self, profiled_folder):
        assert self.has_hypers, "Please register hyper params first. Profiling can only be done after hyper params are registered."
        # list file under the folder
        for file in os.listdir(profiled_folder):
            # end with .csv
            if file.endswith('.csv'):
                # get device name
                target_device = None
                for device_name in self.device_names:
                    if device_name in file:
                        target_device = device_name
                        break
                if target_device is None: continue
                self.profiled_data[target_device] = pd.read_csv(os.path.join(profiled_folder, file))
                # drop the row with lat_avg > 99998
                self.profiled_data[target_device] = self.profiled_data[target_device][self.profiled_data[target_device]['lat_avg'] < 99998]
                self.profiled_data[target_device]['bit'] = self.profiled_data[target_device]['bit'].astype(str)
        
        # check whether each device has one
        for device_name in self.device_names:
            if device_name not in self.profiled_data:
                print(f"Cannot find profiling result for {device_name}, pls add later")

        # read profiled data
        if not self.has_profiled:
            self.has_profiled = True
    
    def predict(self, device_name, shard, b, i, h1, h2, bit):
        assert self.device_in_cost_model(device_name), f"Device {device_name} is not in the cost model."
        if shard == 0:
            model = self.cost_model[device_name][0]
            weight_size, qkv_act_size, kv_concat_size, bmm_act_size, layer_norm_size, dequant_size = mops.SELF_ATTN_MOPS_PARAMS(b, h1, i, bit)
            # predict
            X = np.array([1, weight_size, qkv_act_size, kv_concat_size, bmm_act_size, layer_norm_size, dequant_size])
            # X = sm.add_constant(X)
            return model.predict(X)
        else:
            model = self.cost_model[device_name][1]
            weight_size, act_size, layer_norm_size, dequant_size = mops.FFN_MOPS_PARAMS(b, h1, h2, bit)
            # predict
            X = np.array([1, weight_size, act_size, layer_norm_size, dequant_size])
            # X = sm.add_constant(X)
            return model.predict(X)
    
    def predict_with_hyper(self, device_name, shard, bit):
        assert self.has_hypers, "Hyper parameters are not registered."
        return self.predict(device_name, shard, self.b, self.i, self.h1, self.h2, bit)

    def predict_with_profiled(self, device_name, shard, bit):
        assert self.has_profiled, "Profiled data is not registered."
        return self.fetch_lat_with_hyper(shard, device_name, bit)
