import cv2
import numpy as np
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

KNOWN_WIDTH = 0.5  # Width of human torso 

KNOWN_DISTANCE = 3  # 3 meters away for calibration

cap = cv2.VideoCapture(0)

FOCAL_LENGTH = None

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)

    results = model(frame)

    for detection in results.xyxy[0]:  
        x1, y1, x2, y2, conf, cls = detection
        
        width_in_image = x2 - x1
        height_in_image = y2 - y1
        
        class_name = model.names[int(cls)]
        
        if FOCAL_LENGTH is None:
            FOCAL_LENGTH = (width_in_image * KNOWN_DISTANCE) / KNOWN_WIDTH
            print(f"Calibrated Focal Length: {FOCAL_LENGTH:.2f} pixels")
        
        if FOCAL_LENGTH is not None:
            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / width_in_image

            label = f"{class_name} {conf:.2f} | Distance: {distance:.2f} m"
            dimensions_label = f"Width: {width_in_image:.0f}px | Height: {height_in_image:.0f}px"
            
            frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            frame = cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            frame = cv2.putText(frame, dimensions_label, (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Calibrated Laptop Camera Stream with YOLOv5", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
