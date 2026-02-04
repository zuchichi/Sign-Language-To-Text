# Important Imports
import mediapipe as mp
import cv2
import ultralytics
import time
import torch

# Video capture w/ cv2
cap = cv2.VideoCapture(0)

# Set up mediapipes posing feature
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # Detections
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Recolor frame to BGR to RGB; this is important for MP because they expect an RGB format.

        # Set flag to false
        image.flags.writeable = False # We do not want MP to modify the image's pixel values

        # Detections
        results = hands.process(image)

        # Set flag as true
        image.flags.writeable = True
        
        # Recolor
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        print(results)


        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'): # Hit Q to exit camera
            break

cap.release()
cv2.destroyAllWindows

results.multi_hand_landmarks
print("test")