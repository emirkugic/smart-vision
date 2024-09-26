import cv2
import numpy as np
import requests
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# url = 'http://192.168.0.72:8080/shot.jpg'  
url = 'http://192.168.0.72:8080/shot.jpg'  

KNOWN_WIDTH = 0.15 
FOCAL_LENGTH = 800  

while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, -1)

    results = model(frame)

    for detection in results.xyxy[0]:  
        x1, y1, x2, y2, conf, cls = detection
        width_in_image = x2 - x1
        
        distance = (KNOWN_WIDTH * FOCAL_LENGTH) / width_in_image
        
        label = f"{model.names[int(cls)]} {conf:.2f} | Distance: {distance:.2f} m"
        frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        frame = cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Vision with YOLOv5", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
