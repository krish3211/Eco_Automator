import cv2
import numpy as np
import threading
import time
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class HumanDetection:
    def __init__(self, weights_path, config_path, video_source):
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.output_layers = self.net.getUnconnectedOutLayersNames()
        self.cap = cv2.VideoCapture(video_source)
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = True

        # Define the blue box coordinates
        self.box_x, self.box_y, self.box_w, self.box_h = 270, 190, 100, 100  # Centered box in a 640x480 frame
        self.detected_human = False
        self.last_detection_time = time.time()
        
        # Initialize flag_request as an instance variable
        self.flag_request = "OFF"

        if not self.cap.isOpened():
            raise Exception("Error: Could not open video source.")

    def start_capture(self):
        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.capture_thread.start()

    def capture_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame

    def process_frame(self):
        with self.frame_lock:
            frame = self.latest_frame

        if frame is not None:
            # Draw the blue box
            cv2.rectangle(frame, (self.box_x, self.box_y), (self.box_x + self.box_w, self.box_y + self.box_h), (255, 0, 0), 2)

            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)

            boxes, confidences = [], []
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5 and class_id == 0:  # Human class ID is 0
                        center_x, center_y, w, h = (detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype(int)
                        x, y = int(center_x - w / 2), int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Check if any human is inside the blue box
            self.detected_human = False
            if indexes is not None and len(indexes) > 0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    if (x + w >= self.box_x and x <= self.box_x + self.box_w and
                            y + h >= self.box_y and y <= self.box_y + self.box_h):
                        self.detected_human = True
                        self.last_detection_time = time.time()  # Reset timer
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw the bounding box

            # Print ON or OFF based on detection
            if self.detected_human:
                if self.flag_request == "OFF":
                    self.flag_request = "ON"
                    response = requests.get(os.getenv('ON_TCP'))
                    print(response.status_code)
                    print(self.flag_request)
            else:
                if time.time() - self.last_detection_time > 25:  # 25 seconds
                    if self.flag_request == "ON":
                        self.flag_request = "OFF"
                        response = requests.get(os.getenv('OFF_TCP'))
                        print(response.status_code)
                        print(self.flag_request)

            return frame
        return None

    def stop(self):
        self.running = False
        self.cap.release()
