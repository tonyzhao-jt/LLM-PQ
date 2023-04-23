import numpy as np
from scipy.stats import zipf

# Define the weight matrix
num_rows = 10
num_cols = 5
weights = np.random.randn(num_rows, num_cols)

# Define the Zipf distribution
alpha = 2.0
zipf_dist = zipf(alpha, num_rows)

# Compute the probabilities for each row index
probs = zipf_dist.pmf(np.arange(num_rows))
print(probs)

probs /= probs.sum()

# Sample a row index from the probability distribution
row_idx = np.random.choice(np.arange(num_rows), p=probs)

# Return the sampled row from the weight matrix
sampled_row = weights[row_idx, :]