import cv2
import os

class Config:
    def __init__(self):
        # Get phone camera URL from environment variable or use default
        self.phonecam_url = os.getenv('PHONE_CAMERA_URL', 'http://10.45.7.149:4747/video')
        self.CAMERA_SOURCE = self.select_camera()

    def is_phonecam_available(self):
        """Check if the phone camera stream is accessible"""
        try:
            cap = cv2.VideoCapture(self.phonecam_url)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    return True
            return False
        except Exception as e:
            print(f"Error checking phone camera availability: {str(e)}")
            return False

    def select_camera(self):
        """Automatically select the best available camera"""
        if self.is_phonecam_available():
            print(f"Phone camera detected at {self.phonecam_url}! Using phone camera.")
            return self.phonecam_url
        else:
            print(f"Phone camera unavailable at {self.phonecam_url}. Falling back to webcam.")
            return "webcam"

    def update_camera_source(self):
        """Re-check and update the camera source dynamically"""
        new_source = self.select_camera()
        if new_source != self.CAMERA_SOURCE:
            print(f"Camera source changed to: {new_source}")
            self.CAMERA_SOURCE = new_source
