import os.path
import cv2 # koristimo da bi zaustavili sliku da se ne gasi

from ultralytics import YOLO

#

images_path = ("/home/eldar/PycharmProjects/object_detection_yolo/chapter_5_running_yolo/images/5.png")
yolo_path = ('yolo_weights/yolov8l.pt')


# provjeravamo da li je dobra putanja
if not os.path.exists(images_path):
    print(f"Error: {images_path} does not exist")
elif not os.path.exists(yolo_path):
    print(f"Error: {yolo_path} does not exist")
else:
    model = YOLO(yolo_path)
    results = model(images_path, show = True);


# 'nano'-> yolov8n.pt (ovo je yolo verzija 8 nano .pt) ima jos :
# 'medium' 'large'


cv2.waitKey(0) # 0 znaci ako user nista ne uradi ne radi nista (zastavljamo sliku)


