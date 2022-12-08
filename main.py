#!/usr/bin/env python
# coding: utf-8

# In[7]:

import os
import sys

if'--help' in sys.argv:
    print(  '--data-path\t Caminho para o dataset, o arquivo .yaml deve estar no diretório raiz do dataset\n'\
            + '--batch-size\t (Opicional) Quantas imagens serão carregadas a cada nova época\n'\
            + '--epochs\t (Opcional) Quantas iterações a rede neural irá rodar\n'\
            + '--train\t (Opcional) Se irá treinar ou detectar a rede neural\n'\
            + '--results_path\t (Opcional) Caminho em que será armazenado o resultado do treinamento\n')
    exit(0)

import torch
import glob
import cv2 as cv
import matplotlib.pyplot as plt
from roboflow import Roboflow

import subprocess

TREINAR = True


# In[2]:


def criar_dir_resultados():
    count = len(glob.glob('runs/train/*'))
    print(f'Número de diretórios de resultados: {count}')
    if TREINAR == 'True':
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
    detect_cmd = [ 'python3', 'detect.py', '--weights',\
                f'runs/train/{DIR_RESULTADOS}/weights/best.pt',\
                '--source', data_path, '--name', DIR_DETECCAO]
    subprocess.run(detect_cmd)
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

def treinar(args):
    train_cmd = ['python3', 'train.py','--data', args["data-path"]+'/data.yaml',\
                    '--weights','yolov5m.pt', '--img', '640', '--epochs',\
                    f'{args["epochs"]}', '--batch-size', f'{args["batch-size"]}',\
                    '--name', args["results-path"], '--cache']
    try:
        subprocess.run(train_cmd)
    except RuntimeError as erro:
        if 'CUDA out of memory' in erro:
            torch.cuda.empty_cache()
            args["batch-size"] -= 1
            treinar(args)
        else: return False
    else:
        return True

# In[3]:

os.chdir('yolov5')

if not os.path.exists('yolov5') and not('yolov5' in os.getcwd()):
    subprocess.call(['git', 'clone','https://github.com//ultralytics/yolov5.git'])
    os.system('ls')
    os.chdir('yolov5')
    subprocess.call(['pip3','install', '-r','requirements.txt'])

"""
rf = Roboflow(api_key="nc0bgygPzfvks88x2Dsv")
project = rf.workspace("ic-xo5gl").project("dados_rpg")
dataset = project.version(1).download("yolov5")
"""
if not os.path.exists('Mask-Wearing-4'):
    rf = Roboflow(api_key="nc0bgygPzfvks88x2Dsv")
    project = rf.workspace("joseph-nelson").project("mask-wearing")
    dataset = project.version(4).download("yolov5")


# In[8]:
# Recebendo valores passados pelo shell 
valid_args = {  "--data-path",\
                "--batch-size",\
                "--epochs",\
                "--results-path",\
                "--train"}
arg_values = {"data-path": '', "batch-size": 10, "epochs": 100, "results-path": '', "train": True}
try:
    for index,arg in enumerate(sys.argv):
        if arg in valid_args:
            arg_values[arg[2:]] = sys.argv[index+1]
            print(arg)
except:
    print('No valid arguments')

print(arg_values)
TREINAR = arg_values["train"]
DIR_RESULTADOS = criar_dir_resultados() if arg_values["results-path"] == '' else arg_values["results-path"]
if TREINAR == 'True':
    treinar(arg_values)
    mostrar_resultados(DIR_RESULTADOS)

# In[5]:

IMAGE_INFER_DIR = detectar(DIR_RESULTADOS, arg_values["data-path"]+'/test/images')
visualizar(IMAGE_INFER_DIR)


# In[ ]:




