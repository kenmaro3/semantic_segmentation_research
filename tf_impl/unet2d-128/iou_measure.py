import cv2 as cv
import numpy as np
import tensorflow as tf
from unet2d_mobile_ultra_thin import UNet2D
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

# 幅
W = 640
# 高さ
H = 480
# FPS（Frame Per Second：１秒間に表示するFrame数）
CLIP_FPS = 20.0

output_filepath = 'overlapped.mp4'
output_filepath_masked = 'masked.mp4'
output_filepath_masked_inverse = 'masked_inverse.mp4'
codec = cv.VideoWriter_fourcc('m', 'p', '4', 'v') 
video = cv.VideoWriter(output_filepath, codec, CLIP_FPS, (W, H))
video_masked = cv.VideoWriter(output_filepath_masked, codec, CLIP_FPS, (W, H))
video_masked_inverse = cv.VideoWriter(output_filepath_masked_inverse, codec, CLIP_FPS, (W, H))


stats_list = []

whole_count = cap.get(cv.CAP_PROP_FRAME_COUNT)

ret = True
count = 0
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

    model.load_weights(f"model_128_1m.h5")

    res_dir = "./eval_result"

    res = model.predict(frame)
    res = np.where(res < 0.5, 0, 1)
    res_inverse = np.where(res < 0.5, 1, 0)

    masked = res[0] * frame[0]
    masked_inverse = res_inverse[0] * frame[0]

    # print(res.shape)
    # data = frame[0]*255.0
    # data = data.astype(np.uint8)
    res_dir = "./tmp"
    #plt.imsave(f'{res_dir}/res_{count}.png', tmp[:, :, 0], cmap='gray')
    #plt.imsave(f'{res_dir}/label_{count}.png', label[0], cmap='gray')
    res_cv = cv.cvtColor((res[0]*255).astype(np.float32), cv.COLOR_GRAY2BGR)
    frame = cv.cvtColor((frame[0]*255).astype(np.float32), cv.COLOR_RGB2BGR)
    masked_bgr = cv.cvtColor((masked*255).astype(np.float32), cv.COLOR_RGB2BGR)
    masked_inverse_bgr = cv.cvtColor((masked_inverse*255).astype(np.float32), cv.COLOR_RGB2BGR)
    cv.imwrite(f"{res_dir}/res_{count}.png", res_cv) 
    cv.imwrite(f"{res_dir}/frame_{count}.png", frame) 
    cv.imwrite(f"{res_dir}/masked_{count}.png", masked_bgr) 
    cv.imwrite(f"{res_dir}/masked_inverse_{count}.png", masked_inverse_bgr) 
    img_merged = cv.addWeighted(frame, 0.8, res_cv, 0.2, 0.0, dtype=cv.CV_32F)
    img_merged_resize = cv.resize(img_merged, dsize=(640, 480))
    masked_bgr_resize = cv.resize(masked_bgr, dsize=(640, 480))
    masked_inverse_bgr_resize = cv.resize(masked_inverse_bgr, dsize=(640, 480))
    video.write(np.uint8(img_merged_resize))
    video_masked.write(np.uint8(masked_bgr_resize))
    video_masked_inverse.write(np.uint8(masked_inverse_bgr_resize))

    stats = calculate_stat_helper(res[0][:, :, 0], label[0])
    stats_list.append(stats)

    df = pd.DataFrame(stats_list, columns=['precision', 'recall', 'fvalue', 'iou'])
    count += 1
# print(df.head())
    df.to_csv(f"{res_dir}/res.csv")


video.release()
video_masked.release()
video_masked_inverse.release()