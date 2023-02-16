import json
import os
import sys
import glob
import subprocess
from roboflow import Roboflow
#from torchvision.models import MobileNetV3, MobileNet_V3_Small_Weights
#from keras.applications import MobileNetV3Large
#from keras.models import Sequential
#from keras.callbacks import ModelCheckpoint


class YOLOv7:
    def __init__(self, weights='yolov7x_training.pt'):
        self.weights = weights
        self.dir = os.getcwd()
        print(self.dir)

        if not os.path.exists('yolov7'):
            subprocess.run(['git', 'clone', 'https://github.com/WongKinYiu/yolov7'])
            subprocess.run(['pip', '-r', 'install', f'{self.dir}/yolov7.requirements.txt'])

        if self.weights == None:
            self.weights = '\'\''
        else:
            os.system(
                f'wget -nc -P {self.dir}/yolov7/weights https://github.com/WongKinYiu/yolov7/releases/download/v0.1/{self.weights}')

    def train(self, **kwargs):
        self.training_args = kwargs
        cmd = ['python',  f'{self.dir}/yolov7/train.py']
        for item, value in kwargs.items():
            cmd.append(f'--{item}')
            cmd.append(f'{value}')
        print(cmd)
        subprocess.run(cmd)
    
    def detect(self, **kwargs):
        cmd = ['python', f'{self.dir}/yolov7/detect.py']
        for item, value in kwargs.items():
            cmd.append(f'--{item}')
            cmd.append(f'{value}')
        print(cmd)
        subprocess.run(cmd)


class MobileNet:
    def __init__(self):
        #self.model = MobileNetV3Large(weights='None', classes=2)
        #self.model.compile()
        return

    def train(self, **kwargs):
        return


def getDatasetFromRoboflow(model):
    folderPath = ''
    if model == 'YOLOv7':
        folderName = 'Mask-Wearing-19-YOLOv7'
        folderPath = f'{os.getcwd()}/{folderName}'
        if not os.path.exists(folderName):
                rf = Roboflow(api_key="nc0bgygPzfvks88x2Dsv")
                project = rf.workspace("joseph-nelson").project("mask-wearing")
                dataset = project.version(19).download("yolov7", folderPath)
    if model == 'COCO':
        folderName = 'Mask-Wearing-19-COCO'
        folderPath = f'{os.getcwd()}/{folderName}'
        if not os.path.exists(folderName):
                rf = Roboflow(api_key="nc0bgygPzfvks88x2Dsv")
                project = rf.workspace("joseph-nelson").project("mask-wearing")
                dataset = project.version(19).download("coco", folderPath)
    return folderPath

def convert_coco_to_yolo(coco_annotation_file, yolo_annotation_file):
    with open(coco_annotation_file, 'r') as f:
        coco_annotations = json.load(f)

    yolo_annotations = []
    for image in coco_annotations['images']:
        image_id = image['id']
        for annotation in coco_annotations['annotations']:
            if annotation['image_id'] == image_id:
                x, y, w, h = annotation['bbox']
                x_center = x + w / 2
                y_center = y + h / 2
                class_id = annotation['category_id']
                yolo_annotations.append(
                    f"{class_id} {x_center} {y_center} {w} {h}")

    with open(yolo_annotation_file, 'w') as f:
        for line in yolo_annotations:
            f.write(f"{line}\n")
