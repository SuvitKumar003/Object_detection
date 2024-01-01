
from ultralytics import YOLO
import cv2
import cvzone
import math
#cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture("E:\Object_detection\Videos\pexels_videos_2103099 (2160p).mp4")
 #cap.set(6,2000)
#cap.set(4,2000)
fps = cap.get(6)  # Get frames per second
height = cap.get(4)  # Get frame height
width = cap.get(3)  # Get frame width
print(f"FPS: {fps}, Height: {height}, Width: {width}")
new_height = 400
new_width = 400

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
model = YOLO("../Yolo-weights/yolov8l.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag","tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottepplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier","toothbrush"]
while True:
    success, img=cap.read()
    results=model(img, stream=True)
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)s
            w, h=x2-x1,y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h),l=15)
            #bbox=int(x1) , int(y1), int(w), int(h)

       # print(x1,y1,x2,y2)
        #v2.rectangle(img,(x1,y1),(w,h),(255,0,255),3)
# confidance

        conf=math.ceil((box.conf[0])*100/100)
        cls = int(box.cls[0])
        cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(20,y1)),scale=0.6,thickness=1)


        #print(conf)
    cv2.imshow("Image",img)
    cv2.waitKey(1)