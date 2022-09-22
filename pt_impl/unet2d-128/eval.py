import torch
from torch import nn
import pandas as pd

import cv2 as cv
import glob
import numpy as np

from car_video import Car, CarEval
from model_mobile_thin import UNet2D

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader


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

def main(model_save_path, res_folder="res"):
    bs = 1

    device = torch.device('cuda')

    #root_data = "/Users/kmihara/Downloads/video/*.mp4"
    #root_label = "/Users/kmihara/Downloads/video_label/*.mp4"

    root_data = "/home/ubuntu/workdir/nur/semantic-segmentation/data/CAR/training/video/*"
    root_label = "/home/ubuntu/workdir/nur/semantic-segmentation/data/CAR/training/video_label/*"

    paths_data = sorted(glob.glob(root_data))
    paths_label = sorted(glob.glob(root_label))
    print("list of sequentially data is:  ", paths_data)
    print("list of sequentially label is: ", paths_label)


    video_obj_data = {}
    video_obj_label = {}
    train_frames = {}
    val_frames = {}
    test_frames = {}

    for data, label in zip(paths_data, paths_label):
        video_data = cv.VideoCapture(data)
        video_label = cv.VideoCapture(label)
        video_obj_data[data]=video_data
        video_obj_label[label]=video_label
        train_frames[data] = [0, video_label.get(cv.CAP_PROP_FRAME_COUNT) // 2]
        val_frames[data] = [video_label.get(cv.CAP_PROP_FRAME_COUNT) // 2,
                                video_label.get(cv.CAP_PROP_FRAME_COUNT) // 4 * 3]
        test_frames[data] = [video_label.get(cv.CAP_PROP_FRAME_COUNT) // 4 * 3,
                                    video_label.get(cv.CAP_PROP_FRAME_COUNT)]


    valset = CarEval(paths_data, paths_label, video_obj_data, video_obj_label, train_frames)

    
    model = UNet2D(3, 1)
    model = torch.load(model_save_path)
    model = model.to(device)
    model = model.eval()

    valloader = DataLoader(valset, batch_size=bs, num_workers=0, pin_memory=True)

    m = nn.Sigmoid()

    stats_list = []

    count = 0

    for i, (data, label) in enumerate(valloader):
        print(f"count: {count}================")
        data = data.to(device)
        label = label.to(device)
        with torch.no_grad():
            output = model(data)
            output_sigmoid = m(output)
            output_cpu = output_sigmoid.to('cpu').detach().numpy().copy()
            label_cpu = label.to('cpu').detach().numpy().copy()

        label_cpu = np.where(label_cpu < 0.5, 0, 255)
        labels = label_cpu.astype(np.uint8)

        res = np.where(output_cpu < 0.5, 0, 255)
        tmp = res[0]
        tmp = tmp.astype(np.uint8)

        plt.imsave(f'{res_folder}/res_{i}.png', tmp[0], cmap='gray')
        plt.imsave(f'{res_folder}/label_{i}.png', labels[0, 0, :, :], cmap='gray')

        stats_list.append(calculate_stat_helper(output_cpu, label_cpu))

        df = pd.DataFrame(stats_list, columns=['precision', 'recall', 'fvalue', 'iou'])
        df.to_csv(f"{res_folder}/res.csv")
        count += 1


if __name__ == "__main__":
    print("hello, world")
    main("tmp.pt")
