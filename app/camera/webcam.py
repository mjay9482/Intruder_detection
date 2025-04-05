import cv2
import time
import threading

class Webcam:
    def __init__(self, src=0):
        print("Initializing webcam...")
        self.camera = cv2.VideoCapture(src)

        if not self.camera.isOpened():
            print("Error: Cannot open webcam! Check permissions and index.")
            exit()

        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.ret, self.frame = self.camera.read()
        self.lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(target=self.update_frames, daemon=True)
        self.thread.start()

    def update_frames(self):
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                print("Warning: No frame received!")
                time.sleep(0.1)
                continue

            with self.lock:
                self.ret, self.frame = ret, frame

    def get_frame(self):
        with self.lock:
            return self.ret, self.frame

    def stop(self):
        self.running = False
        self.thread.join()
        self.camera.release()