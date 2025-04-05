import cv2

class Config:
    def __init__(self):
        self.phonecam_url = "http://10.45.4.132:4747/video"
        self.CAMERA_SOURCE = self.select_camera()

    def is_phonecam_available(self):
        """Check if the phone camera stream is accessible"""
        cap = cv2.VideoCapture(self.phonecam_url)
        if cap.isOpened():
            cap.release()
            return True
        return False

    def select_camera(self):
        """Automatically select the best available camera"""
        if self.is_phonecam_available():
            print("Phone camera detected! Using phone camera.")
            return self.phonecam_url
        else:
            print("Phone camera unavailable. Falling back to webcam.")
            return "webcam"

    def update_camera_source(self):
        """Re-check and update the camera source dynamically"""
        new_source = self.select_camera()
        if new_source != self.CAMERA_SOURCE:
            print(f"Camera source changed to: {new_source}")
            self.CAMERA_SOURCE = new_source
