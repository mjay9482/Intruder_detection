from flask import Blueprint, Response, render_template, request
from app.camera.webcam import Webcam
from app.camera.phonecam import Phonecam
from app.config import Config
from app.detection.motion import MotionDetector
from app.detection.object_detection import ObjectDetector
import cv2
import threading
import time

video_bp = Blueprint("video", __name__)

config = Config()
camera = Webcam() if config.CAMERA_SOURCE == "webcam" else Phonecam(config.CAMERA_SOURCE)
motion_detector = MotionDetector()
object_detector = ObjectDetector()

latest_frame = None
latest_motion_frame = None
latest_diff_frame = None
latest_object_frame = None
latest_detections = []
lock = threading.Lock()

def capture_frames():
    """Continuously capture frames from the camera and process them."""
    global latest_frame, latest_motion_frame, latest_diff_frame, latest_object_frame, latest_detections

    while True:
        success, frame = camera.get_frame()
        if not success or frame is None:
            print("Error: Could not read frame!")
            time.sleep(0.1)
            continue

        with lock:
            latest_frame = frame.copy()
            # Motion detection
            _, latest_diff_frame, latest_motion_frame = motion_detector.detect_motion(frame)
            # Object detection
            latest_object_frame, latest_detections = object_detector.detect(frame)

def auto_switch_camera():
    """Automatically switch camera if availability changes"""
    global camera
    while True:
        config.update_camera_source()
        new_source = config.CAMERA_SOURCE
        if isinstance(camera, Phonecam) and new_source == "webcam":
            camera = Webcam()
            print("Switched to webcam dynamically!")
        elif isinstance(camera, Webcam) and new_source.startswith("http"):
            camera = Phonecam(new_source)
            print(f"Switched to phone camera dynamically: {new_source}")
        time.sleep(10)

threading.Thread(target=auto_switch_camera, daemon=True).start()

def generate_stream(frame_type="motion"):
    """Yield frames for streaming based on type: 'motion', 'diff', or 'object'."""
    while True:
        with lock:
            if frame_type == "motion":
                frame = latest_motion_frame
            elif frame_type == "diff":
                frame = latest_diff_frame
            elif frame_type == "object":
                frame = latest_object_frame
            else:
                frame = latest_frame

            if frame is None:
                continue

        # Encode the selected frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Could not encode frame!")
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@video_bp.route("/")
def index():
    return render_template("index.html")

@video_bp.route("/set_camera", methods=["POST"])
def set_camera():
    """Switch between webcam and phone camera dynamically."""
    global camera
    source = request.form.get("source")

    if source == "webcam":
        camera = Webcam(0)
        print("Switched to webcam")
    elif source == "phonecam":
        phone_url = config.phonecam_url
        camera = Phonecam(phone_url)
        print(f"Switched to phone camera: {phone_url}")

    return "Camera source updated!"

@video_bp.route('/video_feed')
def video_feed():
    """Stream the processed video with motion detection overlay."""
    return Response(generate_stream(frame_type="motion"), mimetype='multipart/x-mixed-replace; boundary=frame')

@video_bp.route('/diff_feed')
def diff_feed():
    """Stream the difference mask video."""
    return Response(generate_stream(frame_type="diff"), mimetype='multipart/x-mixed-replace; boundary=frame')

@video_bp.route('/object_feed')
def object_feed():
    """Stream the video with object detection overlay."""
    return Response(generate_stream(frame_type="object"), mimetype='multipart/x-mixed-replace; boundary=frame')

@video_bp.route('/get_detections')
def get_detections():
    """Return the latest object detections as JSON."""
    with lock:
        return {'detections': latest_detections}

threading.Thread(target=capture_frames, daemon=True).start()
