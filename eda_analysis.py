import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from collections import Counter

# Set paths
PATH = os.path.join('data')
actions = np.array(os.listdir(PATH))
sequences = 50
frames = 30

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# 1. Dataset Overview
print("\n1. DATASET OVERVIEW")
print("-" * 40)
print(f"Total Actions (Signs): {len(actions)}")
print(f"Actions: {list(actions)}")
print(f"Total Sequences per Action: {sequences}")
print(f"Frames per Sequence: {frames}")

# Calculate total samples
total_samples = len(actions) * sequences
print(f"Total Data Points: {total_samples}")

# 2. Data Shape Analysis
print("\n2. DATA SHAPE ANALYSIS")
print("-" * 40)
sample_path = os.path.join(PATH, actions[0], '0', '0.npy')
sample_data = np.load(sample_path)
print(f"Shape of single frame: {sample_data.shape}")
print(f"Total features per frame: {sample_data.size}")
print(f"Data type: {sample_data.dtype}")

# 3. Value Distribution Analysis
print("\n3. VALUE DISTRIBUTION ANALYSIS")
print("-" * 40)
all_keypoints = []
for action, sequence in product(actions, range(sequences)):
    for frame in range(frames):
        npy = np.load(os.path.join(PATH, action, str(sequence), str(frame) + '.npy'))
        all_keypoints.append(npy.flatten())

all_keypoints = np.array(all_keypoints)
print(f"Min value: {np.min(all_keypoints):.4f}")
print(f"Max value: {np.max(all_keypoints):.4f}")
print(f"Mean value: {np.mean(all_keypoints):.4f}")
print(f"Std deviation: {np.std(all_keypoints):.4f}")
print(f"Median: {np.median(all_keypoints):.4f}")

# 4. Class Distribution
print("\n4. CLASS DISTRIBUTION")
print("-" * 40)
for action in actions:
    print(f"  {action}: {sequences} sequences")

# 5. Feature Analysis
print("\n5. FEATURE ANALYSIS (Landmark breakdown)")
print("-" * 40)
# Assuming MediaPipe holistic: 21 hand landmarks * 3 coords + 33 pose landmarks * 3 + 468 face landmarks * 3
hand_landmarks = 21 * 3 * 2
print(f"Hand landmarks: {hand_landmarks} features")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Value Distribution
axes[0, 0].hist(all_keypoints.flatten(), bins=50, color='steelblue', edgecolor='black')
axes[0, 0].set_title('Distribution of All Keypoint Values')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')

# Plot 2: Mean values per action
means_per_action = []
for action in actions:
    action_data = []
    for sequence in range(sequences):
        for frame in range(frames):
            npy = np.load(os.path.join(PATH, action, str(sequence), str(frame) + '.npy'))
            action_data.append(npy.mean())
    means_per_action.append(np.mean(action_data))

axes[0, 1].bar(actions, means_per_action, color='coral')
axes[0, 1].set_title('Mean Keypoint Value per Action')
axes[0, 1].set_xlabel('Action')
axes[0, 1].set_ylabel('Mean Value')
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot 3: Std deviation per action
stds_per_action = []
for action in actions:
    action_data = []
    for sequence in range(sequences):
        for frame in range(frames):
            npy = np.load(os.path.join(PATH, action, str(sequence), str(frame) + '.npy'))
            action_data.append(npy.std())
    stds_per_action.append(np.std(action_data))

axes[0, 2].bar(actions, stds_per_action, color='mediumseagreen')
axes[0, 2].set_title('Std Deviation per Action')
axes[0, 2].set_xlabel('Action')
axes[0, 2].set_ylabel('Std Deviation')
axes[0, 2].tick_params(axis='x', rotation=45)

# Plot 4: Box plot for sample action
sample_action_data = []
for sequence in range(sequences):
    for frame in range(frames):
        npy = np.load(os.path.join(PATH, actions[0], str(sequence), str(frame) + '.npy'))
        sample_action_data.append(npy.flatten())

axes[1, 0].boxplot(sample_action_data[:100])
axes[1, 0].set_title(f'Box Plot - First 100 samples of "{actions[0]}"')
axes[1, 0].set_ylabel('Keypoint Values')

# Plot 5: Class balance
axes[1, 1].pie([sequences] * len(actions), labels=actions, autopct='%1.1f%%', 
               colors=plt.cm.Set3.colors[:len(actions)])
axes[1, 1].set_title('Class Distribution')

# Plot 6: Variance heatmap
variance_per_action = []
for action in actions:
    action_variances = []
    for sequence in range(min(sequences, 10)):  # Sample 10 for speed
        frame_data = []
        for frame in range(frames):
            npy = np.load(os.path.join(PATH, action, str(sequence), str(frame) + '.npy'))
            frame_data.append(npy)
        if frame_data:
            action_variances.append(np.var(frame_data, axis=0).mean())
    variance_per_action.append(action_variances)

im = axes[1, 2].imshow(np.array(variance_per_action).T, aspect='auto', cmap='viridis')
axes[1, 2].set_title('Variance Heatmap')
axes[1, 2].set_xlabel('Action')
axes[1, 2].set_ylabel('Sequence')
plt.colorbar(im, ax=axes[1, 2])

plt.tight_layout()
plt.savefig('eda_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nEDA visualization saved as 'eda_visualization.png'")