#!/usr/bin/env python
# coding: utf-8

import os
from utils import *
from roboflow import Roboflow

if __name__ == "__main__":
    model = YOLOv7(None)
    #model.train(data='./Mask-Wearing-4', epochs=200, batch_size=15)

    datasetPath = getDatasetFromRoboflow('YOLOv7')

    training_params = {
        'data': datasetPath,
        'batch-size' : 15,
        'epochs': 200,
        'weights': model.weights,
        'hyp': 'data/hyp.scratch.p5.yaml',
        'cfg': 'cfg/training/yolov7.yaml',
    }
    model.train(**training_params)

    inference_params = {
        'weights': model.weights,
        'conf': 0.25,
        'source': '0'
    }

    model.detect(**inference_params)
