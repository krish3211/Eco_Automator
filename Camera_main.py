# Camera_main.py
import cv2
import os
from dotenv import load_dotenv
from Human_detection import HumanDetection

# Load environment variables from .env file
load_dotenv()

def main():
    # Initialize human detection
    detector = HumanDetection("assets/yolov3.weights", "assets/yolov3.cfg", os.getenv('RTSP_URL'))

    detector.start_capture()

    while True:
        frame = detector.process_frame()
        if frame is not None:
            cv2.imshow("Human Detection", frame)

        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            detector.stop()
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
