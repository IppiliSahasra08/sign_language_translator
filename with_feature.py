import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from tensorflow.keras.utils import to_categorical
import joblib

# Configuration
DATA_PATH = "data"
actions = ["hello", "help", "please", "sorry", "thank_you"]
PATH = os.path.join('data')
sequences = 30
frames = 10

label_map = {label: num for num, label in enumerate(actions)}
Y = []
for action in actions:
    for sequence in range(sequences):
        Y.append(label_map[action])
Y = to_categorical(Y).astype(int)

# Feature Extraction Functions
def extract_features(keypoints_sequence):
    """
    Extract meaningful features from keypoint sequence
    """
    features = []
    
    # 1. Temporal features (frame-to-frame changes)
    if len(keypoints_sequence) > 1:
        velocity = np.diff(keypoints_sequence, axis=0)  # Movement between frames
        acceleration = np.diff(velocity, axis=0)       # Change in velocity
        
        features.extend([
            np.mean(velocity, axis=0),
            np.std(velocity, axis=0),
            np.mean(acceleration, axis=0) if len(acceleration) > 0 else np.zeros_like(keypoints_sequence[0]),
            np.std(acceleration, axis=0) if len(acceleration) > 0 else np.zeros_like(keypoints_sequence[0])
        ])
    else:
        features.extend([np.zeros_like(keypoints_sequence[0])] * 4)
    
    # 2. Spatial features
    features.extend([
        np.max(keypoints_sequence, axis=0),   # Max position
        np.min(keypoints_sequence, axis=0),   # Min position
        np.mean(keypoints_sequence, axis=0),  # Center of motion
        np.std(keypoints_sequence, axis=0)    # Spread of motion
    ])
    
    # 3. Hand-specific features (assuming first 21*3 are left hand, next 21*3 are right)
    hand_data = keypoints_sequence.reshape(len(keypoints_sequence), -1, 3)
    
    # Hand landmarks analysis
    left_hand = hand_data[:, :21, :]  # First 21 landmarks
    right_hand = hand_data[:, 21:42, :]  # Next 21 landmarks
    
    # Hand positions relative to body
    features.append(np.mean(left_hand, axis=(0,1)))  # Average left hand position
    features.append(np.mean(right_hand, axis=(0,1))) # Average right hand position
    
    # Hand movement range
    features.append(np.ptp(left_hand, axis=(0,1)))  # Range of left hand motion
    features.append(np.ptp(right_hand, axis=(0,1))) # Range of right hand motion
    
    return np.concatenate([f.flatten() for f in features])

# Extract features for all data
print("\n" + "=" * 60)
print("WITH FEATURE EXTRACTION")
print("=" * 60)

X_features = []
for action, sequence in product(actions, range(sequences)):
    temp = []
    for frame in range(frames):
        npy = np.load(os.path.join(PATH, action, str(sequence), str(frame) + '.npy'))
        temp.append(npy)
    X_features.append(extract_features(np.array(temp)))

X_features = np.array(X_features)
print(f"Feature Shape: {X_features.shape}")

X_train_f, X_test_f, Y_train_f, Y_test_f = train_test_split(
    X_features, Y, test_size=0.2, random_state=42, stratify=np.argmax(Y, axis=1)
)

Y_train_labels_f = np.argmax(Y_train_f, axis=1)
Y_test_labels_f = np.argmax(Y_test_f, axis=1)

# Random Forest with features
rf_model_f = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_f.fit(X_train_f, Y_train_labels_f)
rf_pred_f = rf_model_f.predict(X_test_f)
cm_rf_f = confusion_matrix(Y_test_labels_f, rf_pred_f)

plt.figure(figsize=(8,6))
sns.heatmap(cm_rf_f, annot=True, fmt='d', cmap='Greens',
            xticklabels=actions, yticklabels=actions)
plt.title("Confusion Matrix - Random Forest (With Features)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("cm_rf_features.png", dpi=150)
rf_accuracy_f = accuracy_score(Y_test_labels_f, rf_pred_f)
print(f"\nRandom Forest (with Features) Accuracy: {rf_accuracy_f:.4f}")

# SVM with features
scaler_f = StandardScaler()
X_train_f_scaled = scaler_f.fit_transform(X_train_f)
X_test_f_scaled = scaler_f.transform(X_test_f)

svm_model_f = SVC(kernel='rbf', random_state=42)
svm_model_f.fit(X_train_f_scaled, Y_train_labels_f)
svm_pred_f = svm_model_f.predict(X_test_f_scaled)
svm_accuracy_f = accuracy_score(Y_test_labels_f, svm_pred_f)
print(f"SVM (with Features) Accuracy: {svm_accuracy_f:.4f}")

# Comparison Summary
print("\n" + "=" * 60)
print("ACCURACY SUMMARY (With Features)")
print("=" * 60)

print(f"{'Model':<20} {'Accuracy':<20}")
print("-" * 40)
print(f"{'Random Forest':<20} {rf_accuracy_f:.4f}")
print(f"{'SVM':<20} {svm_accuracy_f:.4f}")

# Save models
joblib.dump(rf_model_f, "rf_model_features.pkl")
joblib.dump(scaler_f, "scaler_features.pkl")
np.save('actions.npy', actions)
print("\nModels saved: rf_model_features.pkl, scaler_features.pkl, actions.npy")