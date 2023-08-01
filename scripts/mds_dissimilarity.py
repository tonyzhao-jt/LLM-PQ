# originalfile_path
import os 
ROOT_DIR = os.environ['ROOT_DIR']
assert os.path.exists(ROOT_DIR), "ROOT_DIR does not exist"
file_name_prefix = "sols_opt_66b_Tesla_V100-SXM2-32GB_2_NVIDIA_A100-SXM4-40GB_2"
folder_path = f"{ROOT_DIR}/scripts/part_strategy/{file_name_prefix}"
available_bits = [3, 4, '8:tc-li', 16]
# 
interested_file_names = [
    "lat_gamma_0.6",
    "lat_gamma_0.8",
    "lat",
    "group_s_256_n_200_gamma_0.6",
    "group_s_256_n_200_gamma_0.8",
    "group_s_256_n_200_gamma_1",
    "group_s_256_n_100_gamma_0.6",
    "group_s_256_n_100_gamma_0.8",
    "group_s_256_n_100_gamma_1",
    "group_gamma_0.6",
    "group_gamma_0.8",
    "group_1",
]
import pickle
# load data
data_dict = {}
test_method = 'shaq'
for key in interested_file_names:
    path = f"{folder_path}{key}.pkl"
    assert os.path.exists(path), f"{path} does not exist"
    with open(path, "rb") as f:
        data = pickle.load(f)
        # get bitassignment
        bit_assignment = data[test_method]['plan']['bit_assignment']
        bit_val = list(bit_assignment.values())
        bit_val = [available_bits.index(val) for val in bit_val]
        data_dict[key] = bit_val

# update keys
key_mapping = {
    "lat_gamma_0.6": "(128, 120 | 200)",
    "lat_gamma_0.8": "(128, 160 | 200)",
    "lat": "(128, 180 | 200)",
    "group_s_256_n_200_gamma_0.6": "(256, 120 | 200)",
    "group_s_256_n_200_gamma_0.8": "(256, 160 | 200)",
    "group_s_256_n_200_gamma_1": "(256, 200 | 200)",
    "group_s_256_n_100_gamma_0.6": "(256, 60 | 100)",
    "group_s_256_n_100_gamma_0.8": "(256, 80 | 100)",
    "group_s_256_n_100_gamma_1": "(256, 100 | 100)",
    "group_gamma_0.6": "(512, 60 | 100)",
    "group_gamma_0.8": "(512, 80 | 100)",
    "group_1": "(512, 100 | 100)",
}
# update keys
data_dict_new = {}
for key in data_dict:
    data_dict_new[key_mapping[key]] = data_dict[key]
data_dict = data_dict_new

# calculate dissimilarity
# KL divergence of bitwidth distribution
# [1, 0, 1, 2], [1, 0, 1, 2], etc.
# use hamming and cosine
import numpy as np
from scipy.spatial.distance import cosine
from scipy.spatial.distance import hamming
# kl divergence for a list
def kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

# calculte distance between different keys(pairwise)
keys = list(data_dict.keys())
keys.sort()
# calculate distance
distance_matrix = np.zeros((len(keys), len(keys)))
for i in range(len(keys)):
    for j in range(len(keys)):
        key_i = keys[i]
        key_j = keys[j]
        distance_matrix[i, j] = hamming(data_dict[key_i], data_dict[key_j])
# print
print(distance_matrix)

# use MDS to visualize
from sklearn.manifold import MDS
embedding = MDS(n_components=2, dissimilarity='precomputed')
X_transformed = embedding.fit_transform(distance_matrix)
# plot
import matplotlib.pyplot as plt
plt.scatter(X_transformed[:, 0], X_transformed[:, 1])
for i in range(len(keys)):
    plt.text(X_transformed[i, 0], X_transformed[i, 1], keys[i])
plt.show()




