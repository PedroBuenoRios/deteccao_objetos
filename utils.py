import json
import os
import sys
import glob
import subprocess
from roboflow import Roboflow
import datetime

class Detector:
    def __init__(self, preTrained):
        self.workspace = os.getcwd()
        self.training_workspace = self.workspace
        self.detections_workspace = self.workspace 
        self.preTrained = preTrained

    def train(self):
        pass

    def detect(self):
        pass

    def deploy(self):
        pass

class YOLOv7(Detector):
    def __init__(self, preTrained:str='yolov7'):
        super().__init__(preTrained)
        self.training_workspace = f'{self.workspace}/yolov7/runs/trains'
        self.detections_workspace = f'{self.workspace}/yolov7/runs/detections'
        # Clona o repositório do yolov7 se ele não está presente no diretório atual
        if not os.path.exists('yolov7'):
            subprocess.run(['git', 'clone', 'https://github.com/WongKinYiu/yolov7'])
            subprocess.run(['pip3', 'install','-r', f'{self.workspace}/yolov7.requirements.txt'])
        os.chdir('yolov7')

    def callCMDSubprocess(self, arq, args):
        cmd = ['python', arq]
        for item, value in args.items():
            if type(value) != bool:
                cmd.append(f'--{item}')
                cmd.append(f'{value}')
            elif value:
                cmd.append(f'--{item}')
        subprocess.run(cmd)

    def train(self, args:dict):
        self.callCMDSubprocess('train.py', args)
    
    def detect(self, args: dict):
        self.callCMDSubprocess('detect.py', args)

class YOLOv6(Detector):
    def __init__(self, preTrained='yolov6m'):
        super().__init__(preTrained)
        self.workspace = f'{self.workspace}/yolov6'
        self.training_workspace = f'{self.workspace}/runs/trains/'
        self.detections_workspace = f'{self.workspace}/runs/detections/'
       
        if not os.path.exists('./yolov6'):
            subprocess.run(['git', 'clone', 'https://github.com/meituan/yolov6'])
            subprocess.run(['pip3', 'install', '-r', f'{self.workspace}/requirements.txt'])
        os.chdir('yolov6')

    def callCMDSubprocess(self, arq: str, args: dict):
        cmd = ['python', arq]
        for item, value in args.item():
            if type(value) != bool:
                cmd.append(f'--{item}')
                cmd.append(f'{value}')
            elif value:
                cmd.append(f'--{item}')
        subsprocess.run(cmd)

    def train(self, args:dict):
        self.callCMDSubprocess('tools/train.py', args)

    def detect(self, args: dict):
        self.callCMDSubprocess('toolo/detect.py', args)

def convertCOCOToYOLO(jsonPath: str):
    pass


def getDatasetFromRoboflow(model):
    folderPath = ''
    if model == 'YOLOv7':
        folderName = 'Mask-Wearing-19-YOLOv7'
        folderPath = f'{os.getcwd()}/{folderName}'
        if not os.path.exists(folderName):
                rf = Roboflow(api_key="nc0bgygPzfvks88x2Dsv")
                project = rf.workspace("joseph-nelson").project("mask-wearing")
                dataset = project.version(19).download("yolov7", folderPath)
    if model == 'YOLOv6':
        folderName = 'Mask-Wearing-19-YOLOv6'
        folderPath = f'{os.getcwd()}/{folderName}'
        if not os.path.exists(folderName):
                rf = Roboflow(api_key="nc0bgygPzfvks88x2Dsv")
                project = rf.workspace("joseph-nelson").project("mask-wearing")
                dataset = project.version(19).download("mt-yolov6", folderPath)
    if model == 'COCO':
        folderName = 'Mask-Wearing-19-COCO'
        folderPath = f'{os.getcwd()}/{folderName}'
        if not os.path.exists(folderName):
                rf = Roboflow(api_key="nc0bgygPzfvks88x2Dsv")
                project = rf.workspace("joseph-nelson").project("mask-wearing")
                dataset = project.version(19).download("coco", folderPath)
    return folderPath
