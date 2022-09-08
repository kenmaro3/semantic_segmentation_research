import cv2 as cv
import numpy as np
import tensorflow as tf
from unet2d import UNet2D
import pandas as pd
import matplotlib.pyplot as plt
import time
import dl_eval
from PIL import Image
import os
import shutil


def calculate_stat_helper(predict, label):
    test1 = predict
    test2 = label

    tp = np.sum(np.logical_and(test1 >= 0.5, test2 >= 0.5))
    tn = np.sum(np.logical_and(test1 < 0.5, test2 < 0.5))
    fp = np.sum(np.logical_and(test1 >= 0.5, test2 < 0.5))
    fn = np.sum(np.logical_and(test1 < 0.5, test2 >= 0.5))
    iou = tp/(tp+fp+fn)
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    fvalue = 2/((1/precision)+(1/recall))

    return precision, recall, fvalue, iou


test_path = "/home/ubuntu/workdir/nur/semantic-segmentation/data/CAR/validation/video/eval0.mp4"
label = "/home/ubuntu/workdir/nur/semantic-segmentation/data/CAR/validation/video_label/eval0_bin.mp4"
cap = cv.VideoCapture(test_path)
cap_label = cv.VideoCapture(label)


## Model preparation
## Capture camera at location 0
start_frame = 100
skip = 29
cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
cap_label.set(cv.CAP_PROP_POS_FRAMES, start_frame)


stats_list = []

whole_count = cap.get(cv.CAP_PROP_FRAME_COUNT)

ret = True
while ret:
    for _ in range(skip):
        _, frame = cap.read()
        _, frame_label = cap_label.read()

    current_frame = cap.get(cv.CAP_PROP_POS_FRAMES)
    print(f"{current_frame} / {whole_count}========================")

    ret, frame = cap.read()
    ret, label = cap_label.read()

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = cv.resize(frame, dsize=(128, 128))
    # frame = np.array(frame,dtype=np.float32)
    frame = frame/255.
    frame = frame.astype(np.float32)

    label = cv.cvtColor(label, cv.COLOR_BGR2GRAY)
    ret_, label = cv.threshold(label, 50, 255, cv.THRESH_BINARY)
    label = cv.resize(label, dsize=(128, 128))
    label = np.where(label < 50.0 , 0, 1)

    frame = frame[np.newaxis, :]
    label = label[np.newaxis, :]

    print(frame.shape)
    print(label.shape)


    ## Generate model
    inputs  = tf.keras.layers.Input(shape=(128, 128, 3))
    model   = UNet2D(n_channels = 3, n_classes = 1)        ##Note: Do we need to simplify this too?
    outputs = model(inputs)
    model  = tf.keras.Model(inputs, outputs)

    model.load_weights(f"model_128.h5")

    res_dir = "./eval_result"

    res = model.predict(frame)
    res = np.where(res < 0.5, 0, 1)
    tmp = res[0]
    tmp = tmp.astype(np.uint8)
    print(tmp.shape)

    print(res.shape)
    data = frame[0]*255.0
    data = data.astype(np.uint8)
    # pil_image=Image.fromarray(data)
    # pil_image.save(f"{res_dir}/pil.png")
    # plt.imsave(f'{res_dir}/res.png', tmp[:, :, 0], cmap='gray')
    # plt.imsave(f'{res_dir}/label.png', label[0], cmap='gray')

    stats = calculate_stat_helper(tmp[:, :, 0], label[0])
    stats_list.append(stats)

    df = pd.DataFrame(stats_list, columns=['precision', 'recall', 'fvalue', 'iou'])
# print(df.head())
    df.to_csv(f"{res_dir}/res.csv")