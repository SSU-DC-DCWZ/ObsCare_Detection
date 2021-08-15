"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""
 
#from _typeshed import Self
import argparse
import sys
import time
import datetime
from pathlib import Path
import os
import errno

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync

class model:
    def __init__(self, classes, camNum):
        self.weights = '../best.pt'
        self.source = str(camNum) # 요구사항 1 수정
        self.imgsz = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.max_det=1000  # maximum detections per image
        self.device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img=True  # show results
        self.save_txt=False # save results to *.txt
        self.save_conf=False  # save confidences in --save-txt labels
        self.save_crop=False  # save cropped prediction boxes
        self.nosave=False  # do not save images/videos
        self.classes=classes # filter by class: --class 0, or --class 0 2 3
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
        self.loadModel()

    @torch.no_grad()

    def loadModel(self):
        self.save_img = not self.nosave and not self.source.endswith('.txt')  # save inference images
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))
        # Directories
        self.save_dir = increment_path(Path(self.project) / self.name, exist_ok= False)  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        # Initialize
        set_logging()
        self.device = select_device(self.device)
        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        self.classify = False

        # Dataloader
    
    #요구사항2 수정
    def start(self):
        if self.webcam:
            self.view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride)
        width = self.dataset.w
        height = self.dataset.h
        now = datetime.datetime.now()
        self.savename = "./data/Recording/" + self.source + "/" + now.strftime('%Y%m%d%H%M%S') + ".mp4"
        try:  # 파일 경로 생성, 경로가 존재 하지 않을 경우 파일 경로 생성
            if not (os.path.isdir("./data/Recording/" + self.source)):
                os.makedirs(os.path.join("./data/Recording/" + self.source))
                print("./data/Recording/0")
        except OSError as e:  # 생성 실패 시 오류 코드 출력
            if e.errno != errno.EEXIST:
                print("Dir error")
            raise
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.savename, codec, 20.0, ((int(width)), (int(height))))
        self.run()

    def stop(self):
        self.running = False
        self.out.release()

    def run(self):
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        t0 = time.time()
        self.running = True
        for path, img, im0s, vid_cap in self.dataset:
            if self.running == False:
                self.stop()
                break
            pred = self.runInference(path, img)
#
            for i, det in enumerate(pred):  # detections per image
                self.detection(i, det, path, img, im0s)
#
                # Print time (inference + NMS)
                if self.c >= 1:
                    self.writeLog(self.s)

                # Stream results
                if self.view_img:
                    self.loadVideo(str(self.p), self.im0)

            now = datetime.datetime.now()
            if now.strftime('%H%M%S') == '205930':  # 일단위 저장을 위해 00시 00분 00초가 되면 스트리밍을 멈추고 재시작
                self.stop()
                self.start()

    def writeLog(self, name):
        print(f'time, camNum, {name}')

    def loadVideo(self, path, image):
        showtime = datetime.datetime.now()
        cv2.putText(image, showtime.strftime('%Y/%m/%d'), (10,470), cv2.FONT_HERSHEY_DUPLEX,0.5,(255,255,255))
        cv2.putText(image, showtime.strftime('%H:%M:%S'), (555,470), cv2.FONT_HERSHEY_DUPLEX,0.5,(255,255,255))
        cv2.putText(image, 'CAM' + str(0), (575,25), cv2.FONT_HERSHEY_DUPLEX,0.7,(255,255,255)) #스트리밍 화면에 시간, 카메라번호 출력
        cv2.imshow(path, image)
        self.out.write(image)
        if cv2.waitKey(1) == 27:
            self.running = False

    def runInference(self, path, img):
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            t1 = time_sync()
            pred = self.model(img,
                         augment=self.augment,
                         visualize=increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.visualize else False)[0]
            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
            t2 = time_sync()
            return pred

    def detection(self, i, det, path, img, im0s):
        if self.webcam:  # batch_size >= 1
            p, self.s, self.im0, frame = path[i], f'{i}: ', im0s[i].copy(), self.dataset.count

        self.p = Path(p)  # to Path
        #save_path = str(self.save_dir / p.name)  # img.jpg
        txt_path = str(self.save_dir / 'labels' / self.p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')  # img.txt
        self.s += '%gx%g ' % img.shape[2:]  # print string
        self.c = 0
        gn = torch.tensor(self.im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], self.im0.shape).round()
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                self.s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to stri   
            # Write results
            for *xyxy, conf, cls in reversed(det):
                if self.save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                    with open(txt_path + '.txt', 'a') as f:
                         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                    plot_one_box(xyxy, self.im0, label=label, color=colors(c, True), line_thickness=self.line_thickness)
