# -*- coding = utf-8 -*-
# @Time : 2022/7/21 20:18
# @Author：dianlong
import json
import os

import numpy as np

import random
def create2019():
    path = r"E:\Datasets\MICCAI_BraTS_2019_Data_Training"
    listDirs = os.listdir(os.path.join(path, "HGG"))
    listDirs = ["HGG/" + file for file in listDirs]
    listDirs2 = (os.listdir(os.path.join(path, "LGG")))
    listDirs2 = ["LGG/" + file for file in listDirs2]
    # listDirs = os.listdir(os.path.join(path))
    training = []
    list1 = listDirs + listDirs2
    random.shuffle(list1)
    id = 0
    for n in list1:
        files = os.listdir(os.path.join(path, n))
        image = []
        for f in files:
            if f.__contains__("seg"):
                label = n + "/" + f
            else:
                image.append(n + "/" + f)
        training.append({"fold": id//67, "image": image, "label": label})
        id += 1
    fileName = 'brats19_folds.json'
    with open(fileName, 'w') as file:
        file.write(json.dumps({"training": training}, indent=2))


def create2021():
    path = r"E:\Datasets\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    listDirs = os.listdir(path)
    test = []
    for n in listDirs:
        files = os.listdir(os.path.join(path, n))
        image = []
        for f in files:
            if f.__contains__("seg"):
                label = n + "/" + f
            else:
                image.append(n + "/" + f)
        test.append({"fold": -1, "image": image,"label":label})
    fileName = 'brats23_folds.json'
    with open(fileName, 'w') as file:
        file.write(json.dumps({"training": test}, indent=2))


def createliver():
    path = r"E:\Workspace\PythonCode\medical-polar-training-main\medical-polar-training-main\datasets\liver\train"
    listDirs = np.array(os.listdir(path))
    np.random.shuffle(listDirs)
    listDirs = listDirs[len(listDirs) // 2:]
    l = len(listDirs)
    test = []
    for n, idx in zip(listDirs, range(l)):
        if n.__contains__("volume"):
            test.append(
                {"fold": int(idx // 3836.2), "image": "train/" + n, "label": "train/" + "segmentation-" + n[7:]})
    fileName = '../../medical-polar-training-main/medical-polar-training-main/json/liver.json'
    with open(fileName, 'w') as file:
        file.write(json.dumps({"training": test}, indent=2))


import shutil

if __name__ == '__main__':
    # path = r"E:\Workspace\PythonCode\medical-polar-training-main\medical-polar-training-main\datasets\liver\train"
    # path2 = r"E:\Workspace\PythonCode\medical-polar-training-main\medical-polar-training-main\datasets\liver\valid"
    # listDirs = np.array(os.listdir(path))
    # listDirs.sort()
    # for n in listDirs:
    #     if int(n.split("-")[1]) <= 15:
    #         shutil.move(os.path.join(path, n), path2)
    #         print(os.path.join(path, n))
    create2021()
