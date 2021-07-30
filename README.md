# README.md

이 프로젝트는 인공지능이 사람의 동작을 학습하고 실시간으로 송출되고있는 영상 속 사람의 낙상을 감지하고 휠체어나 시작장애인의 안내견 또한 감지해 관측자에게 알려주는 프로그램개발 프로젝트입니다.

본 프로젝트는 2021 공개SW개발의 참가 및 참가자의 역량을 기르기 위해 시작되었습니다. 

먼저 영상학습모델로써 yolov5 를 사용했습니다. 

[GitHub - ultralytics/yolov5: YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite](https://github.com/ultralytics/yolov5)

라이센스 : GNU General Public License v3.0

본모델은 이와같은 pip install을 요구합니다.

```python
# base ----------------------------------------
matplotlib>=3.2.2
numpy>=1.18.5
opencv-python>=4.1.2
Pillow
PyYAML>=5.3.1
scipy>=1.4.1
torch>=1.7.0
torchvision>=0.8.1
tqdm>=4.41.0

# logging -------------------------------------
tensorboard>=2.4.1
# wandb

# plotting ------------------------------------
seaborn>=0.11.0
pandas
```

초기 모델 제작시 총 550 개의 이미지를 사용했으며 mAP_0.5는  0.8정도입니다.

본 모델은 COCO dataset을 바탕으로 pre-train된 모델을 전이학습시켰습니다.

![README%20md%20d8a88002a03d424f85993afd424ebc3b/result.png](README%20md%20d8a88002a03d424f85993afd424ebc3b/result.png)

추후 모델개선에 있어서, 필요한 이미지는 클래스당 1500개, 클래스의 인스턴스가 10000개가 될 수 있도록 조정할 예정입니다.

또한 초기모델의 경우 가장 리소스를 많이 사용하는 YOLOv5x를 사용했습니다. 추후 프로그램에 환경에 따라 모델의 크기를 m이나 l로 조정할 수 있습니다.

![README%20md%20d8a88002a03d424f85993afd424ebc3b/testgif.gif](README%20md%20d8a88002a03d424f85993afd424ebc3b/testgif.gif)