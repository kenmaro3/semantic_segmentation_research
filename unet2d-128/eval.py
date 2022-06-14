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

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass



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
cap.set(cv.CAP_PROP_POS_FRAMES, 0)
cap_label.set(cv.CAP_PROP_POS_FRAMES, 0)

## Generate model
inputs  = tf.keras.layers.Input(shape=(128, 128, 3))
model   = UNet2D(n_channels = 3, n_classes = 1)        ##Note: Do we need to simplify this too?
outputs = model(inputs)
model   = tf.keras.Model(inputs, outputs)

model.load_weights(f"model_128.h5")

# converter = tf.lite.TFLiteConverter.from_keras_model(model)

# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]

# tflite_model = converter.convert()

# tflite_model_file = "model_128.tflite"
# tflite_model_file.write_bytes(tflite_model)

# quit()


res_folder = "./result"
if os.path.exists(res_folder):
    shutil.rmtree(res_folder)
    os.mkdir(res_folder)
else:
    os.mkdir(res_folder)

dl = dl_eval.get_dl(batch_size=1)
loop = 50
time_list = []
for i in range(loop):
    print(f"{i} th loop =======================")

    frames, labels = dl[i]
    frames = frames.astype("float32")
    labels = np.where(labels < 0.5, 0, 255)
    labels = labels.astype(np.uint8)

    t1 = time.time()
    res = model.predict(frames)
    t2 = time.time()
    time_list.append(t2-t1)



    res = np.where(res < 0.5, 0, 255)
    tmp = res[0]
    tmp = tmp.astype(np.uint8)
    print(frames.shape)
    frames = frames*255.0
    frames = frames.astype(np.uint8)
    pil_image=Image.fromarray(frames[0])
    # pil_image.save(f"{res_folder}/pil_{i}.png")
    # plt.imsave(f'{res_folder}/res_{i}.png', tmp[:, :, 0], cmap='gray')
    # plt.imsave(f'{res_folder}/label_{i}.png', labels[0], cmap='gray')

    # res_cv = cv.cvtColor(tmp, cv.COLOR_GRAY2BGR)
    # frame = cv.cvtColor(frames[0], cv.COLOR_RGB2BGR)
    # img_merged = cv.addWeighted(frame, 1.0, res_cv, 1.0, 0.1)
    # img_merged_resize = cv.resize(img_merged, dsize=(640, 480))
    # cv.imwrite(f"{res_folder}/merged_{i}.png", img_merged_resize)

    # stat = calculate_stat_helper(tmp, labels[0])
    # print(f"stat: {stat}")

print("time average:")
print(time_list)
print(np.average(time_list[10:]))
