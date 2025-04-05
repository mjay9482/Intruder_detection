import cv2
import time

class Phonecam:
    def __init__(self, url):
        self.url = url
        self.cap = None
        self.retry_count = 5  
        self.open_camera()

    def open_camera(self):
        """Try opening the phone camera stream with retries"""
        for i in range(self.retry_count):
            self.cap = cv2.VideoCapture(self.url)
            if self.cap.isOpened():
                print(f"Phone camera stream opened: {self.url}")
                return True
            else:
                print(f"Attempt {i+1}/{self.retry_count}: Cannot open phone camera stream")
                time.sleep(2)  
        return False

    def get_frame(self):
        """Read frame from phone camera stream"""
        if self.cap is None or not self.cap.isOpened():
            print("Camera is not initialized properly. Reconnecting...")
            self.open_camera()
            return False, None

        success, frame = self.cap.read()
        if not success:
            print("Failed to capture frame from phone camera")
            self.open_camera()
            return False, None

        return True, frame

    def release(self):
        """Release the camera resource"""
        if self.cap:
            self.cap.release()
            print("Camera released")
