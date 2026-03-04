# model_comparison.py
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.utils import to_categorical
from itertools import product
from sklearn.metrics import confusion_matrix
import seaborn as sns
import joblib
import matplotlib.pyplot as plt

DATA_PATH = "data"
actions = ["hello", "help", "please", "sorry", "thank_you"]
PATH = os.path.join('data')

sequences = 30
frames = 10

label_map = {label: num for num, label in enumerate(actions)}
landmarks, labels = [], []

for action, sequence in product(actions, range(sequences)):
    temp = []
    for frame in range(frames):
        npy = np.load(os.path.join(PATH, action, str(sequence), str(frame) + '.npy'))
        temp.append(npy)
    landmarks.append(temp)
    labels.append(label_map[action])

X = np.array(landmarks)
Y = to_categorical(labels).astype(int)

# Flatten for traditional ML
X_flat = X.reshape(X.shape[0], -1)

no_sequences = 30
sequence_length = 30

X_train = []
X_test = []
Y_train = []
Y_test = []

for action_idx, action in enumerate(actions):
    for sequence in range(no_sequences):

        window = []

        for frame_num in range(sequence_length):
            npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            res = np.load(npy_path)
            window.append(res)

        window = np.array(window).flatten()

        # 🔹 SEQUENCE-WISE SPLIT (80% train, 20% test)
        if sequence < 24:
            X_train.append(window)
            Y_train.append(action_idx)
        else:
            X_test.append(window)
            Y_test.append(action_idx)

# Convert to numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
Y_train_labels = Y_train
Y_test_labels = Y_test

print("=" * 60)
print("WITHOUT FEATURE EXTRACTION (Raw Data)")
print("=" * 60)
print(f"Input Shape: {X_flat.shape}")

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train_labels)
rf_pred = rf_model.predict(X_test)
cm_rf = confusion_matrix(Y_test_labels, rf_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=actions, yticklabels=actions)
plt.title("Confusion Matrix - Random Forest (Raw)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("cm_rf_raw.png", dpi=150)
rf_accuracy = accuracy_score(Y_test_labels, rf_pred)
print(f"\nRandom Forest Accuracy: {rf_accuracy:.4f}")
joblib.dump(rf_model, "rf_model.pkl")
np.save('actions.npy', actions)

# SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, Y_train_labels)
svm_pred = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(Y_test_labels, svm_pred)
print(f"SVM Accuracy: {svm_accuracy:.4f}")

print("\nClassification Report (Random Forest):")
print(classification_report(Y_test_labels, rf_pred, target_names=actions))