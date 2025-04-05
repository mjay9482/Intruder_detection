import cv2
import numpy as np

class MotionDetector:
    def __init__(self, varThreshold=50, history=2000, detectShadows=True, noise_thresh=1100):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=detectShadows, varThreshold=varThreshold, history=history
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.noise_thresh = noise_thresh  

    def detect_motion(self, frame):
        if frame is None:
            print("Error: Received None frame in detect_motion()")
            return False, None, None  

        if not isinstance(frame, np.ndarray):
            print("Error: Frame is not a valid NumPy array")
            return False, None, None

        fg_mask = self.bg_subtractor.apply(frame)

        if fg_mask is None:
            print("Error: Foreground mask is None")
            return False, None, None

        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=4)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False

        for cnt in contours:
            if cv2.contourArea(cnt) > self.noise_thresh:
                motion_detected = True
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, 'Motion Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        fg_mask_colored = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

        return motion_detected, fg_mask_colored, frame
