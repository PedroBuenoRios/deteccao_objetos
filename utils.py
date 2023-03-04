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
        return

    def detect(self):
        return

    def deploy(self):
        return

    # Retorna o caminho para o melhor peso, o diretório de treinamento precisa ser passado
    def get_best_weights(self, path:str):
        return f'{path}/weights/best.pt'

    # Retorna o último diretório de treinamento
    def get_last_training_dir(self):
        amt = len(glob.glob(f'{self.training_workspace}/exp*'))
        return f'{self.training_workspace}/exp{amt}'

    # Retorna um diretório novo para treinamento
    def get_new_training_dir(self):
        amt = len(glob.glob(f'{self.training_workspace}/exp*'))
        return f'{self.training_workspace}/exp{amt+1}'

    # Escreve um relatório de um processo    
    def logOutputAndError(process, path:str):
        with open(f'{path}/log.txt', 'x') as fileLog:
            fileLog.write(f'date: {datetime.datetime.now()}\n')
            fileLog.write(f'Command:\n{process.args}\n')
            fileLog.write(f'OutPut:\n{process.stdout}\n')
            fileLog.write(f'Error:\n{process.stderr}\n')
            fileLog.close()

class YOLOv7(Detector):
    def __init__(self, preTrained:str='yolov7.pt'):
        super().__init__(preTrained)
        self.training_workspace = f'{self.workspace}/yolov7/runs/trains'
        self.detections_workspace = f'{self.workspace}/yolov7/runs/detections'
        # Clona o repositório do yolov7 se ele não está presente no diretório atual
        if not os.path.exists('yolov7'):
            subprocess.run(['git', 'clone', 'https://github.com/WongKinYiu/yolov7'])
            subprocess.run(['pip3', '-r', 'install', f'{self.workspace}/yolov7.requirements.txt'])
        # Realiza o download dos pesos para treinamento se necessário
        if self.preTrained == None:
            self.preTrained = '\'\''
        else:
            os.system(
                f'wget -nc -P {self.workspace}/yolov7/weights https://github.com/WongKinYiu/yolov7/releases/download/v0.1/{self.preTrained}')
            self.preTrained = f'{self.workspace}/yolov7/weights/{self.preTrained}'
   
    def train(self, args:dict):
        # Geração do comando para treinamento
        cmd = ['python3',  f'{self.workspace}/yolov7/train.py']
        for item, value in args.items():
            if type(value) != bool:
                cmd.append(f'--{item}')
                cmd.append(f'{value}')
            elif value:
                cmd.append(f'--{item}')
        print(cmd)
        resp = subprocess.run(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        logOutputAndError(resp, self.get_last_training_dir())
    
    def detect(self, **kwargs):
        cmd = ['python3', f'{self.dir}/yolov7/detect.py']
        for item, value in kwargs.items():
            cmd.append(f'--{item}')
            cmd.append(f'{value}')
        print(cmd)
        subprocess.run(cmd)

class YOLOv6(Detector):
    def __init__(self):
        super().__init__(preTrained)
        self.training_workspace = self.workspace + '/yolov6/runs/trains/'
        self.detections_workspace = self.workspace + '/yolov6/runs/detections/'
        self.preTrained = preTrained
        self.batch = 8
        self.epochs = 200
    
    def train(self, **kwargs):
        print(kwargs)

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
