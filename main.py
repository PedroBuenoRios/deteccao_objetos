#!/usr/bin/env python
# coding: utf-8

# In[7]:

import os
import sys

if'--help' in sys.argv:
    print(  '--data-path\t Caminho para o dataset, o arquivo .yaml deve estar no diretório raiz do dataset\n'\
            '--batch-size\t Quantas imagens serão carregadas a cada nova época\n'\
            '--epochs\t Quantas iterações a rede neural irá rodar\n')
    exit(0)

import torch
import glob
import cv2 as cv
import matplotlib.pyplot as plt
from roboflow import Roboflow

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

def treinar(args, batch_max):
    train_cmd = ['python3', 'train.py','--data', args["data_path"]+'/data.yaml',\
                    '--weights','yolov5m.pt', '--img', '640', '--epochs',\
                    f'{args["epochs"]}', '--batch-size', f'{args["batch"]}',\
                    '--name', args["results_path"], '--cache']
    try:
        subprocess.run(train_cmd)
    except RuntimeError as erro:
        if 'CUDA out of memory' in erro:
            torch.cuda.empty_cache()
            args["batch"] -= 1
            i = batch_max/args["batch"]
            for i in range(batch_max):
                treinar(args, batch_max)
        else: return False
    else:
        return True

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

if not os.path.exists('Mask-Wearing-4'):
    rf = Roboflow(api_key="nc0bgygPzfvks88x2Dsv")
    project = rf.workspace("joseph-nelson").project("mask-wearing")
    dataset = project.version(4).download("yolov5")
    print(dataset)

rf = Roboflow(api_key="nc0bgygPzfvks88x2Dsv")
project = rf.workspace("joseph-nelson").project("mask-wearing")
dataset = project.version(4).download("yolov5")

# In[8]:
# Recebendo valores passados pelo shell 
valid_args = {  "data-path": '--data-path',\
                "batch-size": '--batch-size',\
                "epochs": '--epochs',\
                "results-path": '--results_path'}
args_values = {"data-path": '', "batch-size": 10, "epochs": 100, "results_path": ''}
try:
    for index,arg in enumerate(sys.argv):
        if arg in valid_args:
            arg_values[arg] = sys.argv[index+1]
            print(arg)
except:
    print('No valid arguments')

DIR_RESULTADOS = criar_dir_resultados()
if TREINAR:
    subprocess.run(['python3', 'train.py','--data', './Mask-Wearing-4/data.yaml',\
                    '--weights','yolov5m.pt', '--img', '640', '--epochs',\
                    f'{GERACOES}', '--batch-size', f'{BATCH}', '--name', DIR_RESULTADOS, '--cache'])

print(arg_values)
DIR_RESULTADOS = criar_dir_resultados() if arg_values["results_path"] == '' else arg_values["results_path"]
os.system('ls')
treinar(arg_values, arg_values["batch-size"])

# In[5]:


mostrar_resultados(DIR_RESULTADOS)

IMAGE_INFER_DIR = detectar(DIR_RESULTADOS, args_values["data-path"]+'/test/images')
visualizar(IMAGE_INFER_DIR)


# In[ ]:
caminhoAtual = os.getcwd()
os.system(f'scp -r {caminhoAtual}/runs/detect/{IMAGE_INFER_DIR}  pedro@fenix.local:/home/pedro/IC/deteccao_objetos')



