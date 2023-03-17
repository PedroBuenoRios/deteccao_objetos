import os
from utils import *
from roboflow import Roboflow
import torch
import argparse
import yaml
if __name__ == "__main__":
    torch.cuda.empty_cache()
    model = YOLOv6(None)

    datasetPath = getDatasetFromRoboflow('YOLOv6')
    with open('./configs.yaml', 'r') as f:
        configs = yaml.load(f, Loader=yaml.SafeLoader)
        for opt, args in configs['yolov6'].items():
            model.train(args)
'''
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
'''
   # model.train(args.train)
