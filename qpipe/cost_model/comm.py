import pickle
import os 
import numpy as np 


class CommCostModel:
    def __init__(self, comm_cost_model_folder) -> None:
        assert os.path.exists(comm_cost_model_folder), f"Folder {comm_cost_model_folder} does not exist."
        self.cost_model = {}
        for file in os.listdir(comm_cost_model_folder):
            file_path = os.path.join(comm_cost_model_folder, file)
            if 'cost_model' in file and file.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    self.cost_model.update(pickle.load(f))
    
    def print_model_available_keys(self):
        print(self.cost_model.keys())
    
    def predict_comm_time(self, start_rank, end_rank, data_size):
        key = f"{start_rank}_{end_rank}"
        if key not in self.cost_model:
            key = f"{end_rank}_{start_rank}"
        if key not in self.cost_model:
            raise ValueError(f"Cannot find cost model for {key}")
        if start_rank == end_rank:
            return 0
        model = self.cost_model[key]
        poly = np.poly1d(model)
        cost = poly(data_size)
        # return cost in ms
        # what i want is s
        return cost / 1000