from ultralytics import YOLO
import cv2
import numpy as np
import torch

class ObjectDetector:
    def __init__(self):
        # Load YOLOv8 model
        try:
            self.model = YOLO('yolov8n.pt')
        except Exception as e:
            print(f"Error loading YOLO model: {str(e)}")
            print("Trying alternative loading method...")
            # Alternative loading method
            self.model = YOLO('yolov8n.pt', task='detect')
        
        self.conf_threshold = 0.5  # Confidence threshold
        self.classes = self.model.names  # Get class names

    def detect(self, frame):
        """
        Detect objects in the frame using YOLOv8
        Returns: frame with bounding boxes and labels
        """
        if frame is None:
            return None, []

        try:
            # Run YOLOv8 inference
            results = self.model(frame, conf=self.conf_threshold)[0]
            
            # Create a copy of the frame for drawing
            annotated_frame = frame.copy()
            
            # List to store detected objects
            detected_objects = []

            # Process detections
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                
                # Convert coordinates to integers
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Get class name
                class_name = self.classes[int(class_id)]
                
                # Add to detected objects list
                detected_objects.append({
                    'class': class_name,
                    'confidence': score,
                    'bbox': (x1, y1, x2, y2)
                })
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f'{class_name} {score:.2f}'
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            return annotated_frame, detected_objects
        except Exception as e:
            print(f"Error during object detection: {str(e)}")
            return frame, []  # Return original frame and empty detections on error 