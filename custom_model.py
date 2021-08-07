import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import pandas as pd
import datetime
 
# model 생성
model = torch.hub.load('ultralytics/yolov5', 'custom', path = '/home/kiochy/dcwz/yolov5/runs/train/exp21/weights/best.pt', force_reload = True)

# 추측할 data 삽입
img = '/home/kiochy/dcwz/fallvideo2.mp4'
#results = model(img)


# 클래스를 리스트로 추출
def classExtract(results):
    df = results.pandas().xyxy[0] # pandas dataFrame으로 인식된 데이터 추출
    classnum = df[['class']] ##추출된 pandas dataFrame중 class columm 만 추출
    classlist = classnum.values.tolist() ##추출된 class columm을 list로 변환
    classlist2 = np.concatenate(classlist).tolist() ## 2차원 배열을 1차원 배열로 변환 
    return classlist2


# 클래스를 구분해 상황 판단
def classDistinguish(classlist):
    for i in range(0, len(classlist)):
        if classlist[i] == 1 :
            print("red situation \n ") ## 이건 지금 사람이 누워있는 것을 감지한 것
            now = datetime.datetime.now()
            nowDatetime = now.strftime('%Y_%m_%d%H:%M:%S')
            print(nowDatetime,classlist[i],"has occured")
        elif classlist[i] >= 2 :
            print("yellow situation \n") ## 이건 지금 휠체어, 목발, 시각안내견을 감지한 것
            now = datetime.datetime.now()
            nowDatetime = now.strftime('%Y_%m_%d%H:%M:%S')
            print(nowDatetime,classlist[i],"has occured")
           
            
# 실시간 OD
def realtimeObejectDetection(): 
    cap = cv2.VideoCapture(img)

    while cap.isOpened():
        ret, frame = cap.read()

        # Make detections 
        results = model(frame)
        list = classExtract(results)
        classDistinguish(list)
        cv2.imshow('YOLO', np.squeeze(results.render()))

        if cv2.waitKey(1) & 0xFF == ord('q'):
        	break 

    cv2.destroyAllWindows()
