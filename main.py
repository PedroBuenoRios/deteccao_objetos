#!/usr/bin/env python
# coding: utf-8

# In[7]:

import torch
import glob
import cv2 as cv
import matplotlib.pyplot as plt
from roboflow import Roboflow

import os
import subprocess
from numba import cuda
TREINAR = True
GERACOES = 150
BATCH = 15


# In[2]:


def criar_dir_resultados():
    count = len(glob.glob('runs/train/*'))
    os.system('ls')
    print(f'Número de diretórios de resultados: {count}')
    if TREINAR:
        DIR_RESULTADOS = f'resultados_{count+1}'
    else:
        DIR_RESULTADOS = f'resultados_{count}'
    return DIR_RESULTADOS

def mostrar_resultados(DIR_RESULTADOS):
    os.system(f'ls runs/train/{DIR_RESULTADOS}')
    EXP_PATH = f'runs/train/{DIR_RESULTADOS}'
    pred_imagens_validacao = glob.glob(f'{EXP_PATH}/*_pred.jpg')
    print(pred_imagens_validacao)
    for pred_imagem in pred_imagens_validacao:
        imagem = cv.imread(pred_imagem)
        plt.figure(figsize=(19,16))
        plt.imshow(imagem[:,:,::-1])
        plt.axis('off')
        plt.show

def detectar(DIR_RESULTADOS, data_path):
    deteccao_dir_count = len(glob.glob('runs/detect/*'))
    DIR_DETECCAO = f'deteccao_{deteccao_dir_count+1}'
    print(DIR_DETECCAO)
    subprocess.run(['python3', 'detect.py', '--weights',f'runs/train/{DIR_RESULTADOS}/weights/best.pt','--source', data_path, '--name', DIR_DETECCAO])
    return DIR_DETECCAO

def visualizar(DIR_DETECCAO):
    PATH_DETECCAO = f"runs/detect/{DIR_DETECCAO}"
    detect_imagens = glob.glob(f"{PATH_DETECCAO}/*.jpg")
    print(detect_imagens)
    for pred_imagem in detect_imagens:
        imagem = cv.imread(pred_imagem)
        plt.figure(figsize=(19, 16))
        plt.imshow(imagem[:, :, ::-1])
        plt.axis('off')
        plt.show()


# In[3]:

if not os.path.exists('yolov5') or not('yolov5' in os.getcwd()):
    subprocess.call(['git', 'clone','https://github.com//ultralytics/yolov5.git'])
    os.system('ls')
    os.chdir('yolov5')
    subprocess.call(['pip3','install', '-r','requirements.txt'])
"""
rf = Roboflow(api_key="nc0bgygPzfvks88x2Dsv")
project = rf.workspace("ic-xo5gl").project("dados_rpg")
dataset = project.version(1).download("yolov5")
"""
if not os.path.exists('Mask-Wearing-4')
    rf = Roboflow(api_key="nc0bgygPzfvks88x2Dsv")
    project = rf.workspace("joseph-nelson").project("mask-wearing")
    dataset = project.version(4).download("yolov5")
    print(dataset)

# In[8]:

DIR_RESULTADOS = criar_dir_resultados()
if TREINAR:
    subprocess.run(['python3', 'train.py','--data', './Mask-Wearing-4/data.yaml',\
                    '--weights','yolov5m.pt', '--img', '640', '--epochs',\
                    f'{GERACOES}', '--batch-size', f'{BATCH}', '--name', DIR_RESULTADOS, '--cache'])


# In[5]:


mostrar_resultados(DIR_RESULTADOS)

IMAGE_INFER_DIR = detectar(DIR_RESULTADOS, './Mask-Wearing-4/test/images')
visualizar(IMAGE_INFER_DIR)


# In[ ]:
caminhoAtual = os.getcwd()
os.system(f'scp -r {caminhoAtual}/runs/detect/{IMAGE_INFER_DIR}  pedro@fenix.local:/home/pedro/IC/deteccao_objetos')



