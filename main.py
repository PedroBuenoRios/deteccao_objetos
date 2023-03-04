import os
from utils import *
from roboflow import Roboflow
import torch
    
if __name__ == "__main__":
    torch.cuda.empty_cache()
    model = YOLOv7(None)

    datasetPath = getDatasetFromRoboflow('YOLOv7')

    args = YOLOv7_args()
    args.train['data'] = f'{datasetPath}/data.yaml'
    args.train['batch-size'] = 1
    args.train['epochs'] = 1
    args.train['weights'] = model.preTrained
    args.train['hyp'] = './yolov7/data/hyp.scratch.p5.yaml'
    args.train['cfg'] = './yolov7/cfg/training/yolov7.yaml'
    args.train['cache-images'] = False
    args.train['project'] = model.training_workspace
    args.train['name'] = model.get_new_training_dir()

    model.train(args.train)
