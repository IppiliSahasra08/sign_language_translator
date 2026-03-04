import numpy as np
import os
import string
import mediapipe as mp
import cv2
from my_functions import *
import keyboard
import joblib

# ==========================================================
# FEATURE EXTRACTION FUNCTION (DUPLICATED FROM with_feature.py)
# ==========================================================
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

# ==========================================================
# LOAD DATA + MODEL
# ==========================================================

# Use the features-based model
print("Loading feature-based model...")
rf_model = joblib.load("rf_model_features.pkl")
actions = np.load("actions.npy")

# ==========================================================
# INITIALIZE VARIABLES
# ==========================================================

sentence = []
keypoints = []
last_prediction = None
grammar_result = ""
confidence_threshold = 0.65

# ==========================================================
# START CAMERA
# ==========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot access camera.")
    exit()

# ==========================================================
# MEDIAPIPE HOLISTIC MODEL
# ==========================================================

with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75) as holistic:

    while cap.isOpened():

        ret, image = cap.read()
        if not ret:
            break

        # Process image
        image, results = image_process(image, holistic)
        draw_landmarks(image, results)

        # Extract keypoints
        kp = keypoint_extraction(results)
        
        if np.any(kp):
            keypoints.append(kp)

        # Maintain a sliding window of the last 30 frames
        if len(keypoints) > 30:
            keypoints = keypoints[-30:]

        # ==================================================
        # PREDICTION ON SLIDING WINDOW
        # ==================================================

        if len(keypoints) == 30:
            # Convert to numpy
            keypoints_np = np.array(keypoints)

            # Extract features (30 frames → feature vector)
            features_input = extract_features(keypoints_np).reshape(1, -1)

            # Predict probabilities
            prediction = rf_model.predict_proba(features_input)

            confidence = np.max(prediction)
            predicted_class = np.argmax(prediction)
            predicted_action = actions[predicted_class]

            # Apply confidence threshold
            if confidence > confidence_threshold:
                if last_prediction != predicted_action:
                    sentence.append(predicted_action)
                    last_prediction = predicted_action

        # ==================================================
        # DISPLAY LOGIC
        # ==================================================
        if len(sentence) > 7:
            sentence = sentence[-7:]

        if keyboard.is_pressed(' '):
            sentence = []
            keypoints = []
            last_prediction = None
            grammar_result = ""

        if sentence:
            sentence[0] = sentence[0].capitalize()

        # Merge alphabet (if applicable)
        if len(sentence) >= 2:
            if sentence[-1] in string.ascii_letters:
                if sentence[-2] in string.ascii_letters:
                    sentence[-2] = sentence[-2] + sentence[-1]
                    sentence.pop()
                    sentence[-1] = sentence[-1].capitalize()

        display_text = grammar_result if grammar_result else ' '.join(sentence)
        
        textsize = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_X_coord = (image.shape[1] - textsize[0]) // 2

        cv2.putText(image, display_text, (text_X_coord, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Camera - Feature Extraction Model', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.getWindowProperty('Camera - Feature Extraction Model', cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()
