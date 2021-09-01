#from _typeshed import Self
import sys
import time
import datetime
from pathlib import Path
import os
import errno

import cv2
import torch
import torch.backends.cudnn as cudnn
from DB_video import videoDB
from DB_log import logDB

import threading

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path


# 본 프로젝트는 YOLOv5 및 deepSORT를 바탕으로 object detection model을 custom train 시킨 모델을 사용합니다.
# YOLOv5 와 deepSORT의 라이브러리 함수들을 import해 fall detection 및 specific obeject detection 및 alert 에 필요한 parameter를 가져올 수 있게 합니다.
#mog
from utils.google_utils import attempt_download
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadWebcam
from utils.general import check_img_size, check_imshow,non_max_suppression, scale_coords, xyxy2xywh,set_logging, increment_path
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device,time_sync 
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

def compute_color_for_id(label):
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

class model:
    def __init__(self, classes, camNum):
        self.weights = '../best.pt'  # 학습시킨 모델을 weights로 가져옵니다.
        self.source = str(camNum) #재생될 source 본 프로젝트에서는 웹캠입니다.
        self.imgsz = 640 # 이미지 사이즈
        self.conf_thres = 0.45 # 추측임계값
        self.iou_thres = 0.45 #iou임계값
        self.max_det=1000  # maximum detections per image
        self.device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img=True  # show results
        self.save_txt=False # save results to *.txt
        self.save_conf=False  # save confidences in --save-txt labels
        self.save_crop=False  # save cropped prediction boxes
        self.nosave=False  # do not save images/videos
        self.classes= classes # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False  # augmented inference
        self.visualize=False  # visualize features
        self.update=False  # update all models
        self.project='./runs/detect'  # save results to project/name
        self.name='exp'  # save results to project/name
        self.exist_ok=False  # existing project/name ok, do not increment
        self.line_thickness=3  # bounding box thickness (pixels)
        self.hide_labels=False  # hide labels
        self.hide_conf=False  # hide confidences
        self.half=False
        self.running = False

        self.fallTimeList = [] #falltime 시간 측정시 사용되는 리스트
        self.objectTimeList = [] #objecttime 측정 사물 측정시 사용되는 리스트

        self.id = None
        self.fallId = None
        self.objectId = None
        self.noti = None

        self.loadModel() #생상자 생성때 loadModel함수를 가져와


    @torch.no_grad()
    # 모델을 로드합니다. 
    def loadModel(self):
        self.webcam = self.source.isnumeric()
        # Initialize  
        set_logging()
        self.device = select_device(self.device)
        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        self.classify = False
        cfg = get_config()
        cfg.merge_from_file("/home/kiochy/dcwz/yolov5/deep_sort_pytorch/configs/deep_sort.yaml")
        #attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    
    def start(self):
        if self.webcam:
            cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset =  LoadStreams(self.source, img_size=self.imgsz, stride=self.stride) # 모델이 적용될 영상을 받아오게 합니다.
        width = self.dataset.w  # 영상의 길이
        height = self.dataset.h #영상의 높이
        fps = self.dataset.fps[0] # 영상의 fps
        now = datetime.datetime.now()  # 현재시간 가져오기
        self.starttime = datetime.datetime.now() 
        self.savename = "./data/Recording/" + self.source + "/" + now.strftime('%Y%m%d%H%M%S') + ".mp4" # 영상 저장 
        try:  # 파일 경로 생성, 경로가 존재 하지 않을 경우 파일 경로 생성
            if not (os.path.isdir("./data/Recording/" + self.source)):
                os.makedirs(os.path.join("./data/Recording/" + self.source))
        except OSError as e:  # 생성 실패 시 오류 코드 출력
            if e.errno != errno.EEXIST:
                print("Dir error")
            raise
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.savename, codec, fps, ((int(width)), (int(height))))
        self.run()

    def stop(self):
        self.running = False
        self.out.release()
        db = videoDB.DBvideo(self.source, self.starttime, self.savename)
        db.makerecord()
        del db

    def run(self):
        # Run inference
        if self.device.type != 'cpu': #  GPU 설정일 때 
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.running = True

        #dataset에서 path, img, im0s, vid_cap인자를 가져옵니다. 본 프로젝트에서는 웹캠으로 실시간으로 dataset을 업데이트 해 인자들을 가져옵니다.
        for path, img, im0s, vid_cap in self.dataset:
            if self.running == False:
                self.stop()
                break

            pred = self.runInference(path, img) #runInference에서 pred값을 가져옵니다.

            for i, det in enumerate(pred):  # detections per image
                self.detection(i, det, path, img, im0s)
                showtime = datetime.datetime.now()
                cv2.putText(self.im0, showtime.strftime('%Y/%m/%d'), (10,710), cv2.FONT_HERSHEY_DUPLEX,0.5,(255,255,255))
                cv2.putText(self.im0, showtime.strftime('%H:%M:%S'), (1200,710), cv2.FONT_HERSHEY_DUPLEX,0.5,(255,255,255))
                cv2.putText(self.im0, 'CAM' + str(0), (1200,25), cv2.FONT_HERSHEY_DUPLEX,0.7,(255,255,255)) #스트리밍 화면에 시간, 카메라번호 출력
                
                if self.c == 1:
                    self.falldetection()

                
                if self.c == 0:
                    self.objectdetection()

                # Stream results
                if self.view_img:
                    self.loadVideo(str(self.p))

            now = datetime.datetime.now()
            if now.strftime('%H%M%S') == '000000':  # 일단위 저장을 위해 00시 00분 00초가 되면 스트리밍을 멈추고 재시작
                self.stop()
                self.start()

    def falldetection(self):
        if self.fallId is None:
            self.fallId = self.id
        elif self.fallId != self.id:
            #print("1")
            return
        else:
            #print("2")
            now = datetime.datetime.now()
            self.fallTimeList.append(now)

            if len(self.fallTimeList) >= 2 :
                time = self.fallTimeList[-1] - self.fallTimeList[0]
            else:
                time = datetime.timedelta(0, 0, 0, 0 , 0 ,0, 0) 
            if int(time.total_seconds()) >= 6:
                self.fallTimeList = []
            #print(time.total_seconds())
            if int(time.total_seconds()) == 5: ##연속적 falldetect
                print("fall is detected")
                print(datetime.datetime.now())
                self.fallTimeList = [] ## 시간 초기화

    def objectdetection(self):
        if self.objectId is None:
            self.objectId = self.id
        elif self.objectId != self.id:
            self.noti = None
            return
        else:
            if self.objectId == self.id and self.noti == None:
                print("detected")
                self.noti = 1
            #now = datetime.datetime.now() 
            #self.objectTimeList.append(now)
            #
            #if self.prev is not None:
            #    if int(now.total_seconds() - self.prev.total_seconds()) <= 10:
            #        return
            #    else:
            #        del(self.prev)
    #
            #if len(self.list) >= 2:
            #    time = self.list[-1] - self.list[0]
    #
            #else:
            #    time = datetime.timedelta(0, 0, 0, 0, 0, 0, 0)
    #
            #if int(time.total_seconds()) >= 2:
            #    print(f'{self.c} is detected')
            #    self.screenshot(self.c)
            #    self.list = []  # 시간 초기화
            #    self.prev = datetime.datetime.now()
            
        

    def writeLog(self, name):
        print(f'time, camNum, {name}')

    def loadVideo(self, path):
        cv2.imshow(path, self.im0)
        self.out.write(self.im0)
        if cv2.waitKey(1) == 27:
            self.running = False

    def screenshot(self, situation):
        now = datetime.datetime.now()
        path = './data/Situation/' + str(situation) + '/' + now.strftime('%Y%m%d%H%M%S_' + str(situation)) + '.jpg'
        try:  # 파일 경로 생성, 경로가 존재 하지 않을 경우 파일 경로 생성
            if not (os.path.isdir("./data/Situation/" + str(situation))):
                os.makedirs(os.path.join("./data/Situation/" + str(situation)))
        except OSError as e:  # 생성 실패 시 오류 코드 출력
            if e.errno != errno.EEXIST:
                print("Dir error")
            raise
        cv2.imwrite(path, self.im0)
        im = logDB.DBlog(now, situation, self.source, path)
        im.makerecord()
        del im


    #영상을 프레임 단위로 잘라 이미지 파일로 추론을 진행합니다. 이중 가장 추론율이 높은 바운딩 박스의 좌표값을 리턴합니다.
    def runInference(self, path, img):
        img = torch.from_numpy(img).to(self.device) #웹캠으로부터 가져온 이미지를 numpy형식으로 가져옵니다.
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3: #차원을 추가해 바운딩 박스가 들어갈 차원을 생성합니다.
            img = img.unsqueeze(0)
        
        # Inference : 이미지를 가져와 바운딩 박스를 그립니다.
        pred = self.model(img,
                     augment=self.augment,
                     visualize=increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.visualize else False)[0]
        # Apply NMS(예측한 바운딩 박스 중 가장 정확도가 높은 박스를 선택합니다.)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        return pred


    #추론된 바운딩박스값을 출력될 영상에 올바른 위치와 클래스값으로 출력될 수 있게 합니다.
    def detection(self, i, det, path, img, im0s):
        if self.webcam:  # batch_size >= 1
            p, self.s, self.im0, frame = path[i], f'{i}: ', im0s[i].copy(), self.dataset.count

        self.p = Path(p)  # to Path
        self.s += '%gx%g ' % img.shape[2:]  # print string
        self.c = 0
        if len(det):
            # Rescale boxes from img_size to im0 size 추론한 이미지의 사이즈를 실제 출력될 영상 사이즈에 맞게 리스케일 합니다.
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], self.im0.shape).round()
            # Print results 
            for self.c in det[:, -1].unique():
                n = (det[:, -1] == self.c).sum()  # detections per class
                self.s += f"{n} {self.names[int(self.c)]}{'s' * (n > 1)}, "  # add to string
            
            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]

            outputs = self.deepsort.update(xywhs.cpu(), confs.cpu(), clss, self.im0)

            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, confs)): 
                    
                    bboxes = output[0:4]
                    self.id = output[4]
                    cls = output[5]
                    self.c = int(cls)  # integer class
                    label = f'{self.id} {self.names[self.c]} {conf:.2f}'
                    color = compute_color_for_id(self.id)
                    plot_one_box(bboxes, self.im0, label=label, color=color, line_thickness=2)
                        


                        

                    
            ####dasd# Write results
            ####dasdfor *xyxy, conf, cls in reversed(det):
            ####dasd    #if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
            ####dasd    self.c = int(cls)  # integer class
            ####dasd    label = None if self.hide_labels else (self.names[self.c] if self.hide_conf else f'{self.names[self.c]} {conf:.2f}') #출력될 레이블의 이름
            ####dasd    plot_one_box(xyxy, self.im0, label=label, color=colors(self.c, True), line_thickness=self.line_thickness) #이미지 위에 출력될 바운딩 박스를 생성합니다.
                
