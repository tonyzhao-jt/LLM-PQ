import os 
import numpy as np 
import pickle
import pandas as pd 
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib
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
    def __init__(self, device_names=[], lat_cost_model_folder=None) -> None:
        self.device_names = device_names
        self.has_hypers = False
        self.has_profiled = False
        self.has_fit = False
        self.profiled_data = {}
        self.regression_models = {}

        self.profiled_prepost_data = {}
        self.has_fit_prepost = False

    def device_in_cost_model(self, device_name):
        return device_name in self.device_names
    
    def register_hyper_params(self, b, s, i, h1, h2):
        self.b = b
        self.s = s # prompt length
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
    
    def fetch_lat(self, device_name, shard, b, s, i, h1, h2, bit):
        profiled_data_device = self.profiled_data[device_name]
        # columns: shard,h1,h2,bit,batch_size,input_seq_length,past_seq_length,lat_avg,mem_weight,mem_kv,mem_embedding,mem_all
        # fetch data with hyper params
        profiled_data_device = profiled_data_device[(profiled_data_device['shard'] == shard) &
                                            (profiled_data_device['h1'] == h1) &
                                            (profiled_data_device['h2'] == h2) &
                                            (profiled_data_device['bit'] == str(bit)) &
                                            (profiled_data_device['batch_size'] == b) &
                                            (profiled_data_device['input_seq_length'] == s) &
                                            (profiled_data_device['past_seq_length'] == i)]
        if len(profiled_data_device) == 0:
            return None
        # lat_avg
        lat_avg = profiled_data_device['lat_avg'].values[0]
        return lat_avg
    
    # pure profiler
    # fatch prefill result
    # s is the prompt length
    def fetch_prefill(self, s, shard, device_name, bit):
        # input_seq = s
        # past_seq = 0
        i = 0
        return self.fetch_lat(device_name, shard, self.b, s, i, self.h1, self.h2, bit)
    
    def fetch_prefill_use_hyper_s(self, shard, device_name, bit):
        s = self.s 
        return self.fetch_prefill(s, shard, device_name, bit)

    # fetch decoding result
    # i is the past sequence length
    def fetch_decode(self, i, shard, device_name, bit):
        # input_seq = s
        # past_seq = s
        s = 1 # one token
        return self.fetch_lat(device_name, shard, self.b, s, i, self.h1, self.h2, bit)
    
    def fetch_decode_use_hyper_i(self, shard, device_name, bit):
        i = self.i
        return self.fetch_decode(i, shard, device_name, bit)
    
    # following are pure analytical model
    def update_profiled_result(self, profiled_folder):
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
                if target_device not in self.profiled_data:
                    self.profiled_data[target_device] = pd.read_csv(os.path.join(profiled_folder, file))
                else:
                    # update the pandas array
                    self.profiled_data[target_device] = self.profiled_data[target_device]._append(pd.read_csv(os.path.join(profiled_folder, file)))
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
    
        # following are pure analytical model
    
    def update_profiled_prepost_result(self, profiled_folder):
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
                if target_device not in self.profiled_prepost_data:
                    self.profiled_prepost_data[target_device] = pd.read_csv(os.path.join(profiled_folder, file))
                else:
                    # update the pandas array
                    self.profiled_prepost_data[target_device] = self.profiled_prepost_data[target_device]._append(pd.read_csv(os.path.join(profiled_folder, file)))
                # drop the row with lat_avg > 99998
                self.profiled_prepost_data[target_device] = self.profiled_prepost_data[target_device][self.profiled_prepost_data[target_device]['time'] < 99998]
        
        # check whether each device has one
        for device_name in self.device_names:
            if device_name not in self.profiled_prepost_data:
                print(f"Cannot find profiling result for {device_name}, pls add later")

        # read profiled data
        if not self.has_fit_prepost:
            self.has_fit_prepost = True

    def fetch_prepost_lat(self, device_name, stage, batch_size, prompt_length):
        # model_name,model_size,h1,h2,batch_size,prompt_length,stage,time
        # input_seq = s
        # past_seq = 0
        h1, h2 = self.h1, self.h2
        profiled_data = self.profiled_prepost_data[device_name]
        profiled_data_device = profiled_data[(profiled_data['prompt_length'] == prompt_length) &
                                            (profiled_data['batch_size'] == batch_size) &
                                            (profiled_data['stage'] == stage) &
                                            (profiled_data['h1'] == h1) &
                                            (profiled_data['h2'] == h2)]
        # lat_avg
        lat_avg = profiled_data_device['time'].values[0]
        return lat_avg


    def verbose_regression_names(self):
        assert len(self.regression_models) > 0, "Please load regression models first."
        for device_name in self.device_names:
            print(f"Device {device_name} has {len(self.regression_models[device_name])} regression models.")
            # print(f"Regression models are: {self.regression_models[device_name].keys()}")

    def load_regression_cost_model(self, cost_model_store_path="./leanred_cost_model"):
        assert len(self.profiled_data) > 0, "Please update profiled result first."
        all_mse = []
        for device_name in self.device_names:
            self.regression_models[device_name] = {}
            device_profile_data = self.profiled_data[device_name]
            # generate a fitted result for each shard, each h1 h2, bit. leaving batch_size and past_seq_length as variables
            # columns: shard,h1,h2,bit,batch_size,input_seq_length,past_seq_length,lat_avg,mem_weight,mem_kv,mem_embedding,mem_all
            # for each shard, h1, h2, bit, we fit a model
            shards = device_profile_data['shard'].unique()
            h_pairs = device_profile_data[['h1', 'h2']].drop_duplicates()
            bits = device_profile_data['bit'].unique()
            input_seq_lengths = device_profile_data['input_seq_length'].unique()
            for shard in shards:
                for h1, h2 in h_pairs.values:
                    for bit in bits:
                        for input_seq_length in input_seq_lengths:
                            model_name = f"{device_name}_{shard}_{h1}_{h2}_{input_seq_length}_{bit}.pkl"
                            # load model if exists
                            if os.path.exists(os.path.join(cost_model_store_path, model_name)):
                                self.regression_models[device_name][model_name] = model = joblib.load(os.path.join(cost_model_store_path, model_name))
                                # verify the model if has data
                                profiled_data_device = device_profile_data[(device_profile_data['shard'] == shard) & 
                                                                    (device_profile_data['h1'] == h1) &
                                                                    (device_profile_data['h2'] == h2) &
                                                                    (device_profile_data['input_seq_length'] == input_seq_length) &
                                                                    (device_profile_data['bit'] == str(bit))]

                                X = profiled_data_device[['batch_size', 'past_seq_length']].values
                                X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
                                y = profiled_data_device['lat_avg'].values
    
                                y_pred = model.predict(X)
                                mse = mean_squared_error(y, y_pred)
                                all_mse.append(mse)
                                # print(f'MSE: {mse:.3f}')
                                # print(f'Intercept: {model.intercept_}')
                                # print(f'Coefficients: {model.coef_}')
                            else:
                                print(f"Cannot find regression model {model_name} for device {device_name}. pls run fit")
                                # raise Exception(f"Cannot find regression model {model_name} for device {device_name}. pls run fit")
        print("Verified all regress models, maximum mse: ", max(all_mse))
        self.has_fit = True

    def fit_regression_cost_model(self, cost_model_store_path='./leanred_cost_model', store=True):
        # mkfir
        if not os.path.exists(cost_model_store_path):   
            os.mkdir(cost_model_store_path)
        # start fitting...
        # according our usage, we only care about the past_length's impact on the latency
        # when others are fixed.
        assert len(self.profiled_data) > 0, "Please update profiled result first."
        
        for device_name in self.device_names:
            device_profile_data = self.profiled_data[device_name]
            # generate a fitted result for each shard, each h1 h2, bit. leaving batch_size and past_seq_length as variables
            # columns: shard,h1,h2,bit,batch_size,input_seq_length,past_seq_length,lat_avg,mem_weight,mem_kv,mem_embedding,mem_all
            # for each shard, h1, h2, bit, we fit a model
            shards = device_profile_data['shard'].unique()
            h_pairs = device_profile_data[['h1', 'h2']].drop_duplicates()
            input_seq_lengths = device_profile_data['input_seq_length'].unique() # important, =1 then decode, >1 then prefill
            bits = device_profile_data['bit'].unique()
            for shard in shards:
                for h1, h2 in h_pairs.values:
                    for bit in bits:
                        for input_seq_length in input_seq_lengths:
                            model_name = f"{device_name}_{shard}_{h1}_{h2}_{input_seq_length}_{bit}.pkl"
                            print(model_name)
                            # fetch data with hyper params
                            profiled_data_device = device_profile_data[(device_profile_data['shard'] == shard) & 
                                                                    (device_profile_data['h1'] == h1) &
                                                                    (device_profile_data['h2'] == h2) &
                                                                    (device_profile_data['input_seq_length'] == input_seq_length) &
                                                                    (device_profile_data['bit'] == str(bit))]
                            if len(profiled_data_device) == 0:
                                print(f"Cannot find profiled data for {model_name}, skip")
                                continue
                            # fit a model
                            # columns: shard,h1,h2,bit,batch_size,input_seq_length,past_seq_length,lat_avg,mem_weight,mem_kv,mem_embedding,mem_all
                            # X: batch_size, past_seq_length
                            # y: lat_avg
                            X = profiled_data_device[['batch_size', 'past_seq_length']].values
                            X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
                            y = profiled_data_device['lat_avg'].values
                            model = LinearRegression().fit(X, y)
                            # store the model
                            if device_name not in self.regression_models:
                                self.regression_models[device_name] = {}
                            self.regression_models[device_name][model_name] = model
                            # store the model
                            y_pred = model.predict(X)
                            mse = mean_squared_error(y, y_pred)
                            print(f'MSE: {mse:.4f}')
                            # print(y - y_pred)
                            # print(f'Intercept: {model.intercept_}')
                            # print(f'Coefficients: {model.coef_}')
                            if store:
                                joblib.dump(model, os.path.join(cost_model_store_path, f'{model_name}'))

        self.has_fit = True

    def predict(self, device_name, shard, b, s, i, h1, h2, bit):
        assert self.has_fit, "Cost model is not fitted."
        assert device_name in self.regression_models, f"Cannot find regression model for {device_name}"
        model_name = f"{device_name}_{shard}_{h1}_{h2}_{s}_{bit}.pkl"
        if model_name not in self.regression_models[device_name]:
            # print(f"Cannot find regression model for {model_name}")
            return None
        model = self.regression_models[device_name][model_name]
        X = np.array([[b, i, 1]])
        return model.predict(X)[0]
    
    def predict_same_bit(self, device_name, b, s, i, h1, h2, bit):
        assert self.has_fit, "Cost model is not fitted."
        assert device_name in self.regression_models, f"Cannot find regression model for {device_name}"
        shard = 2
        model_name = f"{device_name}_{shard}_{h1}_{h2}_{s}_{bit}.pkl"
        if model_name not in self.regression_models[device_name]:
            # print(f"Cannot find regression model for {model_name}")
            return None
        model = self.regression_models[device_name][model_name]
        X = np.array([[b, i, 1]])
        return model.predict(X)[0]
    
    def predict_same_bit_with_b_s_i_bit(self, device_name, b, s, i, bit):
        return self.predict_same_bit(device_name, b, s, i, self.h1, self.h2, bit)
    
    def predict_same_bit_by_profiled_with_b_s_i_bit(self, device_name, b, s, i, bit):
        shard = 2
        return self.fetch_lat(device_name, shard, b, s, i, self.h1, self.h2, bit)

    def predict_by_model_with_b_s_i_bit(self, device_name, shard, b, s, i, bit):
        return self.predict(device_name, shard, b, s, i, self.h1, self.h2, bit)

    def predict_by_profiled_with_b_s_i_bit(self, device_name, shard, b, s, i, bit):
        return self.fetch_lat(device_name, shard, b, s, i, self.h1, self.h2, bit)
    
    # only use hyper to predict.
    def predict_with_hyper(self, device_name, shard, bit):
        assert self.has_hypers, "Hyper parameters are not registered."
        return self.predict(device_name, shard, self.b, self.i, self.h1, self.h2, bit)

    def predict_with_profiled(self, device_name, shard, bit):
        assert self.has_profiled, "Profiled data is not registered."
        return self.fetch_lat_with_hyper(shard, device_name, bit)


    