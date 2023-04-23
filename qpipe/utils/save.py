import os, pickle
def save_with_pickle(result, file_name=None, folder_path=None):
    assert file_name is not None, "file_name should not be None"
    # if folder path not exists, create one
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    result_path = os.path.join(folder_path, file_name)
    with open(result_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"Result saved to {result_path}")