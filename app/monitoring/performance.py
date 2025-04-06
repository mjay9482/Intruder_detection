import time
import threading
from collections import deque
import numpy as np
from datetime import datetime, timedelta

class PerformanceMonitor:
    def __init__(self, window_size=100):
        # FPS tracking
        self.frame_times = deque(maxlen=window_size)
        self.last_frame_time = time.time()
        
        # Latency tracking
        self.motion_latencies = deque(maxlen=window_size)
        self.object_latencies = deque(maxlen=window_size)
        
        # Camera switching tracking
        self.camera_switches = deque(maxlen=window_size)
        self.switch_start_time = None
        
        # Detection accuracy tracking
        self.motion_detections = deque(maxlen=window_size)
        self.object_detections = deque(maxlen=window_size)
        self.false_positives = deque(maxlen=window_size)
        self.total_detections = 0
        self.total_false_positives = 0
        
        # System uptime tracking
        self.start_time = time.time()
        self.downtime = 0
        self.last_error_time = None
        self.error_duration = 0
        
        # Connection tracking
        self.connection_failures = 0
        self.connection_recoveries = 0
        self.last_recovery_time = None
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_metrics, daemon=True)
        self.monitoring_thread.start()

    def update_frame_time(self):
        """Update frame timing metrics"""
        current_time = time.time()
        with self.lock:
            self.frame_times.append(current_time - self.last_frame_time)
            self.last_frame_time = current_time

    def update_motion_latency(self, latency):
        """Update motion detection latency"""
        with self.lock:
            self.motion_latencies.append(latency)

    def update_object_latency(self, latency):
        """Update object detection latency"""
        with self.lock:
            self.object_latencies.append(latency)

    def start_camera_switch(self):
        """Start tracking camera switch time"""
        self.switch_start_time = time.time()

    def end_camera_switch(self):
        """End tracking camera switch time"""
        if self.switch_start_time:
            with self.lock:
                self.camera_switches.append(time.time() - self.switch_start_time)
            self.switch_start_time = None

    def update_motion_detection(self, detected, is_true_positive):
        """Update motion detection accuracy metrics"""
        with self.lock:
            self.motion_detections.append(1 if detected else 0)
            if detected:
                self.total_detections += 1
                if not is_true_positive:
                    self.total_false_positives += 1
                    self.false_positives.append(1)
                else:
                    self.false_positives.append(0)

    def update_object_detection(self, detections, ground_truth):
        """Update object detection accuracy metrics"""
        with self.lock:
            if detections and ground_truth:
                # Calculate IoU for each detection
                ious = []
                for det in detections:
                    max_iou = 0
                    for gt in ground_truth:
                        iou = self._calculate_iou(det['bbox'], gt['bbox'])
                        max_iou = max(max_iou, iou)
                    ious.append(max_iou)
                
                # Consider detection correct if IoU > 0.5
                correct_detections = sum(1 for iou in ious if iou > 0.5)
                self.object_detections.append(correct_detections / len(detections))
                
                # Update false positive tracking
                self.total_detections += len(detections)
                false_positives = len(detections) - correct_detections
                self.total_false_positives += false_positives
                self.false_positives.append(false_positives / len(detections) if len(detections) > 0 else 0)
            elif detections:
                self.object_detections.append(0.5)  # Assume 50% accuracy as a baseline
                # Consider all detections as potential false positives when no ground truth
                self.total_detections += len(detections)
                self.total_false_positives += len(detections)
                self.false_positives.append(1.0)  

    def record_connection_failure(self):
        """Record a connection failure"""
        with self.lock:
            self.connection_failures += 1
            self.last_error_time = time.time()
            # Start tracking downtime
            if self.last_error_time:
                self.error_duration = time.time() - self.last_error_time
                self.downtime += self.error_duration

    def record_connection_recovery(self):
        """Record a successful connection recovery"""
        with self.lock:
            self.connection_recoveries += 1
            self.last_recovery_time = time.time()
            # End tracking downtime
            if self.last_error_time:
                self.error_duration = 0
                self.last_error_time = None

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / (box1_area + box2_area - intersection)

    def _monitor_metrics(self):
        """Continuously monitor and log metrics"""
        while True:
            time.sleep(1)  # Update every second
            self._log_metrics()

    def _log_metrics(self):
        """Calculate and log current metrics"""
        with self.lock:
            # Calculate FPS
            fps = 1.0 / np.mean(self.frame_times) if self.frame_times else 0
            
            # Calculate latencies
            motion_latency = np.mean(self.motion_latencies) * 1000 if self.motion_latencies else 0
            object_latency = np.mean(self.object_latencies) * 1000 if self.object_latencies else 0
            
            # Calculate camera switch time
            switch_time = np.mean(self.camera_switches) * 1000 if self.camera_switches else 0
            
            # Calculate detection accuracy
            motion_accuracy = np.mean(self.motion_detections) * 100 if self.motion_detections else 0
            object_map = np.mean(self.object_detections) * 100 if self.object_detections else 0
            
            # Calculate false positive reduction
            if self.total_detections > 0:
                false_positive_rate = (self.total_false_positives / self.total_detections) * 100
            else:
                false_positive_rate = 0
            false_positive_reduction = max(0, 100 - false_positive_rate)
            
            # Calculate uptime
            total_time = time.time() - self.start_time
            current_downtime = self.downtime
            if self.last_error_time:
                current_downtime += time.time() - self.last_error_time
            uptime = ((total_time - current_downtime) / total_time) * 100
            
            # Calculate connection recovery rate
            recovery_rate = (self.connection_recoveries / self.connection_failures * 100) if self.connection_failures > 0 else 0
            
            # Log metrics
            print(f"\nPerformance Metrics:")
            print(f"FPS: {fps:.1f}")
            print(f"Motion Detection Latency: {motion_latency:.1f}ms")
            print(f"Object Detection Latency: {object_latency:.1f}ms")
            print(f"Camera Switch Time: {switch_time:.1f}ms")
            print(f"Motion Detection Accuracy: {motion_accuracy:.1f}%")
            print(f"Object Detection mAP: {object_map:.1f}%")
            print(f"False Positive Reduction: {false_positive_reduction:.1f}%")
            print(f"System Uptime: {uptime:.1f}%")
            print(f"Connection Recovery Rate: {recovery_rate:.1f}%")

    def get_metrics(self):
        """Get current metrics as a dictionary"""
        with self.lock:
            # Calculate current downtime including ongoing errors
            current_downtime = self.downtime
            if self.last_error_time:
                current_downtime += time.time() - self.last_error_time
            
            # Calculate false positive reduction
            if self.total_detections > 0:
                false_positive_rate = (self.total_false_positives / self.total_detections) * 100
            else:
                false_positive_rate = 0
            false_positive_reduction = max(0, 100 - false_positive_rate)
                
            return {
                'fps': 1.0 / np.mean(self.frame_times) if self.frame_times else 0,
                'motion_latency': np.mean(self.motion_latencies) * 1000 if self.motion_latencies else 0,
                'object_latency': np.mean(self.object_latencies) * 1000 if self.object_latencies else 0,
                'switch_time': np.mean(self.camera_switches) * 1000 if self.camera_switches else 0,
                'motion_accuracy': np.mean(self.motion_detections) * 100 if self.motion_detections else 0,
                'object_map': np.mean(self.object_detections) * 100 if self.object_detections else 0,
                'false_positive_reduction': false_positive_reduction,
                'uptime': ((time.time() - self.start_time - current_downtime) / (time.time() - self.start_time)) * 100,
                'recovery_rate': (self.connection_recoveries / self.connection_failures * 100) if self.connection_failures > 0 else 0
            } 