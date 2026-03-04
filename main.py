# ==========================================================
# IMPORTS
# ==========================================================
import numpy as np
import os
import string
import mediapipe as mp
import cv2
from my_functions import *
import keyboard
import joblib

# ==========================================================
# LOAD DATA + MODEL
# ==========================================================

# Load trained Random Forest model and actions
rf_model = joblib.load("rf_model.pkl")
actions = np.load("actions.npy")

# ==========================================================
# INITIALIZE VARIABLES
# ==========================================================

sentence = []
keypoints = []
last_prediction = None
grammar_result = ""
confidence_threshold = 0.5 # Lowered slightly for real-time variance

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
        
        # Only append to keypoints if at least one hand is detected
        # (This prevents filling the buffer with zeros)
        if np.any(kp):
            keypoints.append(kp)
        else:
            # Optional: if you want to clear the buffer when hands are removed
            # keypoints = []
            pass

        # Maintain a sliding window of the last 30 frames
        if len(keypoints) > 30:
            keypoints = keypoints[-30:]

        # ==================================================
        # PREDICTION ON SLIDING WINDOW
        # ==================================================

        if len(keypoints) == 30:
            # Convert to numpy
            keypoints_np = np.array(keypoints)

            # Flatten (30 frames → 1 vector)
            flat_input = keypoints_np.flatten().reshape(1, -1)

            # Predict probabilities
            prediction = rf_model.predict_proba(flat_input)

            confidence = np.max(prediction)
            predicted_class = np.argmax(prediction)
            predicted_action = actions[predicted_class]

            # Apply confidence threshold
            if confidence > confidence_threshold:
                if last_prediction != predicted_action:
                    sentence.append(predicted_action)
                    last_prediction = predicted_action
        else:
            # If no hands or not enough frames, we don't predict
            pass

        # ==================================================
        # LIMIT SENTENCE LENGTH
        # ==================================================

        if len(sentence) > 7:
            sentence = sentence[-7:]

        # ==================================================
        # RESET ON SPACEBAR
        # ==================================================

        if keyboard.is_pressed(' '):
            sentence = []
            keypoints = []
            last_prediction = None
            grammar_result = ""

        # ==================================================
        # CAPITALIZE FIRST WORD
        # ==================================================

        if sentence:
            sentence[0] = sentence[0].capitalize()

        # ==================================================
        # MERGE ALPHABET LETTERS INTO WORDS
        # ==================================================

        if len(sentence) >= 2:
            if sentence[-1] in string.ascii_letters:
                if sentence[-2] in string.ascii_letters:
                    sentence[-2] = sentence[-2] + sentence[-1]
                    sentence.pop()
                    sentence[-1] = sentence[-1].capitalize()

        # ==================================================
        # DISPLAY TEXT
        # ==================================================

        display_text = grammar_result if grammar_result else ' '.join(sentence)

        textsize = cv2.getTextSize(display_text,
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   1, 2)[0]

        text_X_coord = (image.shape[1] - textsize[0]) // 2

        cv2.putText(image,
                    display_text,
                    (text_X_coord, 470),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA)

        # Show camera
        cv2.imshow('Camera', image)

        # Exit if window closed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
            break

# ==========================================================
# CLEANUP
# ==========================================================

cap.release()
cv2.destroyAllWindows()