# preprocessing_comparison.py
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import product

PATH = os.path.join('data')
actions = np.array(os.listdir(PATH))
sequences = 30
frames = 10

# Load RAW data
raw_data = []
for action, sequence in product(actions, range(sequences)):
    for frame in range(frames):
        npy = np.load(os.path.join(PATH, action, str(sequence), str(frame) + '.npy'))
        raw_data.append(npy)

raw_data = np.array(raw_data)
print(f"Raw Data Shape: {raw_data.shape}")

# PREPROCESSING TECHNIQUES

# 1. Normalization (Min-Max Scaling)
def normalize_minmax(data):
    return (data - data.min()) / (data.max() - data.min() + 1e-8)

# 2. Standardization (Z-score)
def standardize(data):
    return (data - data.mean()) / (data.std() + 1e-8)

# 3. Relative positioning (normalize to first frame)
def relative_positioning(data):
    return data - data[0:1]

# 4. Temporal differencing
def temporal_difference(data):
    return np.diff(data, axis=0)

normalized_data = normalize_minmax(raw_data)
standardized_data = standardize(raw_data)

# Comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Raw data distribution
axes[0, 0].hist(raw_data.flatten(), bins=50, color='red', alpha=0.7)
axes[0, 0].set_title('RAW Data Distribution')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')

# Normalized data
axes[0, 1].hist(normalized_data.flatten(), bins=50, color='blue', alpha=0.7)
axes[0, 1].set_title('NORMALIZED Data (Min-Max)')
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Frequency')

# Standardized data
axes[1, 0].hist(standardized_data.flatten(), bins=50, color='green', alpha=0.7)
axes[1, 0].set_title('STANDARDIZED Data (Z-Score)')
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Frequency')

# Comparison statistics
stats_data = {
    'Raw': [raw_data.mean(), raw_data.std(), raw_data.min(), raw_data.max()],
    'Normalized': [normalized_data.mean(), normalized_data.std(), normalized_data.min(), normalized_data.max()],
    'Standardized': [standardized_data.mean(), standardized_data.std(), standardized_data.min(), standardized_data.max()]
}

x = np.arange(4)
width = 0.25
metrics = ['Mean', 'Std', 'Min', 'Max']

axes[1, 1].bar(x - width, stats_data['Raw'], width, label='Raw', color='red')
axes[1, 1].bar(x, stats_data['Normalized'], width, label='Normalized', color='blue')
axes[1, 1].bar(x + width, stats_data['Standardized'], width, label='Standardized', color='green')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(metrics)
axes[1, 1].legend()
axes[1, 1].set_title('Statistics Comparison')

plt.tight_layout()
plt.savefig('preprocessing_comparison.png', dpi=150)
plt.show()