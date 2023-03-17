import json
import os
import sys
import glob
import subprocess
from roboflow import Roboflow
import datetime

class YOLOv7_args:
    def __init__(self):
        self.train = {
            'weights' : 'yolov7.pt',
            'cfg' : '',
            'data': '',
            'hyp' : '',
            'epochs' : 100,
            'batch-size' : 15,
            'img-size' : 640,
            'rect': False,
            'resume' : False,
            'nosave' : False,
            'notest' : False,
            'noautoanchor' : False,
            'evolve' : False,
            'bucket' : '',
            'cache-images' : True,
            'image-weights' : False,
            'device' : '',
            'multi-scale' : False,
            'single-cls' : False,
            'adam' : False,
            'sync-bn' : False,
            'local_rank' : -1,
            'workers' : 8,
            'project' : './yolov7/runs/train',
            'entity' : None,
            'name' : './yolov7/runs/train/exp',
            'exist-ok': False,
            'quad' : False,
            'linar-lr' : False,
            'label-smoothing' : 0.0,
            'upload_dataset' : False,
            'bbox_interval' : -1,
            'save_period' : -1,
            'artifact_alias' : 'latest',
            'freeze' : 0,
            'v5-metric': False 
        }
        self.detect = {}

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

    # Retorna o caminho para o melhor peso, o diretório de treinamento precisa ser passado
    def get_best_weights(self, path:str) -> str:
        return f'{path}/weights/best.pt'

    # Retorna o último diretório de treinamento
    def get_last_training_dir(self) -> str:
        amt = len(glob.glob(f'{self.training_workspace}/exp*'))
        return f'{self.training_workspace}/exp{amt}'

    # Retorna um diretório novo para treinamento
    def get_new_training_dir(self) -> str:
        amt = len(glob.glob(f'{self.training_workspace}/exp*'))
        return f'{self.training_workspace}/exp{amt+1}'

class YOLOv7(Detector):
    def __init__(self, preTrained:str='yolov7'):
        super().__init__(preTrained)
        self.training_workspace = f'{self.workspace}/yolov7/runs/trains'
        self.detections_workspace = f'{self.workspace}/yolov7/runs/detections'
        # Clona o repositório do yolov7 se ele não está presente no diretório atual
        if not os.path.exists('yolov7'):
            subprocess.run(['git', 'clone', 'https://github.com/WongKinYiu/yolov7'])
            subprocess.run(['pip3', 'install','-r', f'{self.workspace}/yolov7.requirements.txt'])
        # Realiza o download dos pesos para treinamento se necessário
        if self.preTrained == None:
            self.preTrained = '\'\''
        else:
            subprocess.run(
                ['wget', '-nc', '-P',
                f'{self.workspace}/yolov7/weights', 
                f'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/{self.preTrained}.pt'])
            self.preTrained = f'{self.workspace}/yolov7/weights/{self.preTrained}.pt'
   
    def train(self, args:dict):
        # Geração do comando para treinamento
        cmd = ['python3',  f'{self.workspace}/yolov7/train.py']
        for item, value in args.items():
            if type(value) != bool:
                cmd.append(f'--{item}')
                cmd.append(f'{value}')
            elif value:
                cmd.append(f'--{item}')
        try:
            subprocess.run(cmd)
        except:
            pass
    
    def detect(self, **kwargs):
        cmd = ['python3', f'{self.dir}/yolov7/detect.py']
        for item, value in kwargs.items():
            cmd.append(f'--{item}')
            cmd.append(f'{value}')
        print(cmd)
        subprocess.run(cmd)

class YOLOv6(Detector):
    def __init__(self, preTrained='yolov6m'):
        super().__init__(preTrained)
        self.training_workspace = f'{self.workspace}/yolov6/runs/trains/'
        self.detections_workspace = f'{self.workspace}/yolov6/runs/detections/'
        if not os.path.exists('./yolov6'):
            subprocess.run(['git', 'clone', 'https://github.com/meituan/yolov6'])
            subprocess.run(['pip3', 'install', '-r', f'{self.workspace}/yolov6/requirements.txt'])
        if self.preTrained == None:
            self.preTrained = '\'\''
        else:
            subprocess.run(['wget', '-nc', '-P',
                f'{self.workspace}/yolov6/weights',
                f'https://github.com/meituan/YOLOv6/releases/download/0.3.0/{self.preTrained}.pt'])
            self.preTrained = f'{self.workspace}/yolov6/weights/{self.preTrained}.pt'

    def train(self, args:dict):
        cmd = ['python3', f'{self.workspace}/yolov6/tools/train.py']
        for item,value in args.items():
            if type(value) != bool:
                cmd.append(f'--{item}')
                cmd.append(f'{value}')
            elif value:
                cmd.append(f'--{item}')
            try:
                subprocess.run(cmd)
            except:
                pass

    def detect(self, **kwargs):
        print(kwargs)

    def deploy(self, **kwargs):
        print

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
