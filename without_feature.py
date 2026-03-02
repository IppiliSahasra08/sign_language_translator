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

# Load data
PATH = os.path.join('data')
actions = np.array(os.listdir(PATH))
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

X_train, X_test, Y_train, Y_test = train_test_split(
    X_flat, Y, test_size=0.2, random_state=42, stratify=np.argmax(Y, axis=1)
)

Y_train_labels = np.argmax(Y_train, axis=1)
Y_test_labels = np.argmax(Y_test, axis=1)

print("=" * 60)
print("WITHOUT FEATURE EXTRACTION (Raw Data)")
print("=" * 60)
print(f"Input Shape: {X_flat.shape}")

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train_labels)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(Y_test_labels, rf_pred)
print(f"\nRandom Forest Accuracy: {rf_accuracy:.4f}")

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