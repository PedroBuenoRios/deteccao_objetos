import os
from utils import *
from roboflow import Roboflow
import torch
import argparse
import yaml
if __name__ == "__main__":
    torch.cuda.empty_cache()
    datasetPath = getDatasetFromRoboflow('YOLOv7')
    model = YOLOv7(None)
    with open('../opts.yaml', 'r') as f:
        configs = yaml.load(f, Loader=yaml.SafeLoader)
        for opt, args in configs['yolov7'].items():
            model.train(args)

    infArgs = {
            'weights': f'./runs/train/exp12/weights/best.pt',
            'conf': 0.25,
            'img-size': 480,
            'source': f'{datasetPath}/test/images/*.jpg'
            }

   # model.detect(infArgs)
