"""
Model loading with Pytorch

Video capture and graphics with OpenCV

Object detection with YOLOv5

Object tracking with DeepSORT
"""

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import torch

class YoloDetector():
    """
    This pre-trained model will allow us to recognize objects in frame, as classified in the COCO training dataset:
    https://github.com/ultralytics/yolov5/blob/master/data/coco.yaml

    It will determine the class of the detected object, along with the coordinates of the corresponding bounding box.
    """

    # Initialization
    def __init__(self, model_name):
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Device: ")
        print(self.classes)
    
    # Model loading
    def load_model(self, model_name):
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    
    # Frame forward pass to model
    def score_frame(self, frame):
        self.model.to(self.device)
        downscale_factor = 2
        width = int(frame.shape[1]/downscale_factor)
        height = int(frame.shape[0]/downscale_factor)
        frame = cv2.resize(frame, (width, height))
        
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].to('cpu').numpy(), results.xyxyn[0][:, :-1]

        return labels, cord
    
    # Convert class to label
    def class_to_label(self, x):
        return self.classes[int(x)]
    
    # Box plotting
    def plot_boxes(self, results, frame, height, width, confidence=0.3):
        
        labels, cord = results
        detections = []
        n = len(labels)
        x_shape, y_shape = width, height

        for i in range(n):
            row = cord[i]
            
            if row[4] >= confidence:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                
                if self.class_to_label(labels[i]) in self.classes.values():
                    confidence = float(row[4].item())
                    detections.append(([x1, y1, int(x2-x1), int(y2-y1)], confidence, labels[i], self.classes[labels[i]]))
        
        return frame, detections, labels
    
# Setting webcam video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Calling object detector class
detector = YoloDetector(model_name='yolov5m')

# Calling object tracking class
object_tracker = DeepSort(max_age=20)

while cap.isOpened():
    # Object detection during capture
    success, img = cap.read()
    start = time.perf_counter()
    results = detector.score_frame(img)
    img, detections, labels = detector.plot_boxes(results,
                                  img,
                                  height=img.shape[0],
                                  width=img.shape[1],
                                  confidence=0.5)

    # Object tracking during capture
    tracks = object_tracker.update_tracks(detections, frame=img)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        bbox = ltrb
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        
        try:
            cv2.putText(img, "ID: " + str(track_id) + "    " + str(detections[0][-1]), (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            print(detections)
        except:
            cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Frames per second display
    end = time.perf_counter()
    totalTime = end - start
    fps = 1 / totalTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv2.imshow('img', img)

    # Input ESC key to terminate
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Termination
cap.release()
cv2.destroyAllWindows()

