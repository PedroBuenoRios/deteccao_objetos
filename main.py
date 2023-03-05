# coding: utf-8

import os
from utils import *
import torch
from roboflow import Roboflow
if __name__ == "__main__":
    torch.cuda.empty_cache()
    model = YOLOv7(None)
    #model.train(data='./Mask-Wearing-4', epochs=200, batch_size=15)

    datasetPath = getDatasetFromRoboflow('YOLOv7')

    training_params = {
        'data': datasetPath + '/data.yaml' ,
        'batch-size' : 1,
        'epochs': 100,
        'weights': model.weights,
        'hyp': './yolov7/data/hyp.scratch.p5.yaml',
        'cfg': './yolov7/cfg/training/yolov7.yaml',
        'cache-images': False
    }
    #model.train(**training_params)

    inference_params = {
        'weights': model.weights,
        'conf': 0.25,
        'source': '0'
    }

    model.detect(**inference_params)
