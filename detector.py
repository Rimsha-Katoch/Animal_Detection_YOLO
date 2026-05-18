from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

animal_classes = ["brids", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]

def detect_animals(frame):
    results = model(frame)

    # loop through detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0]) # get class id
            class_name = model.names[class_id] # convert class id into class name

            # check if detected object is animal
            if class_name in animal_classes:
                x1,y1,x2,y2 = map(int, box.xyxy[0]) # get box coordinates
                confidence = float(box.conf[0])
                label = f"{class_name} : {confidence*100:.1f}%"

                # draw rectangle
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

                # put text
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return frame
                


