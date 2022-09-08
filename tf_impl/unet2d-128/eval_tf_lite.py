import tensorflow as tf
import cv2 as cv
import numpy as np
import os
import time
import shutil
from PIL import Image
import dl_eval
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    print("hello, world")
    tf_lite_path = "model_128.tflite"

    interpreter = tf.lite.Interpreter(tf_lite_path)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0] 
    #print(input_details)
    output_details = interpreter.get_output_details()[0]
    #print(output_details)

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


    res_folder = "./result"
    if os.path.exists(res_folder):
        shutil.rmtree(res_folder)
        os.mkdir(res_folder)
    else:
        os.mkdir(res_folder)

    dl = dl_eval.get_dl(batch_size=1)
    loop = 5
    time_list = []
    for i in range(loop):
        print(f"{i} th loop =======================")

        frames, labels = dl[i]
        frames = frames.astype("float32")
        labels = np.where(labels < 0.5, 0, 255)
        labels = labels.astype(np.uint8)
        frames = frames.astype(np.float32)

        t1 = time.time()

        interpreter.set_tensor(input_details['index'], frames)

        interpreter.invoke()
        res = interpreter.get_tensor(output_details['index'])
        #res = model.predict(frames)
        t2 = time.time()
        time_list.append(t2-t1)



        res = np.where(res < 0.5, 0, 255)
        tmp = res[0]
        tmp = tmp.astype(np.uint8)
        print(frames.shape)
        frames = frames*255.0
        frames = frames.astype(np.uint8)
        pil_image=Image.fromarray(frames[0])
        pil_image.save(f"{res_folder}/pil_{i}.png")
        plt.imsave(f'{res_folder}/res_{i}.png', tmp[:, :, 0], cmap='gray')
        plt.imsave(f'{res_folder}/label_{i}.png', labels[0], cmap='gray')

        res_cv = cv.cvtColor(tmp, cv.COLOR_GRAY2BGR)
        frame = cv.cvtColor(frames[0], cv.COLOR_RGB2BGR)
        img_merged = cv.addWeighted(frame, 1.0, res_cv, 1.0, 0.1)
        img_merged_resize = cv.resize(img_merged, dsize=(640, 480))
        cv.imwrite(f"{res_folder}/merged_{i}.png", img_merged_resize)

        stat = calculate_stat_helper(tmp, labels[0])
        print(f"stat: {stat}")

    print("time average:")
    print(time_list)
    print(np.average(time_list[10:]))
