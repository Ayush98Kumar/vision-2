import cv2
from ultralytics import YOLO

class FaceMultiDetector:
    def __init__(self):
        # Ultralytics auto-downloads the model if not found
        self.model = YOLO("yolov8l.pt")  # standard YOLOv8 nano, detects 'person' class (class 0)

    def process_frame(self, frame):
        # Detect only 'person' class (class 0) as face-level proxy. Reduced imgsz for speed.
        results = self.model.predict(frame, classes=[0], conf=0.7, verbose=False, imgsz=320)
        
        for r in results:
            if len(r.boxes) > 1:
                return "Multiple Faces Detected"
        return None

if __name__ == "__main__":
    detector = FaceMultiDetector()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        result = detector.process_frame(frame)
        if result:
            print(f"Flag: {result}")
            
        cv2.imshow("Face Cheat Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
