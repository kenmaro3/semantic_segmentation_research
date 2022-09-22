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

device = torch.device('cuda')

model_save_path = "model_128_ultra_thin.pt"
res_folder = "res_thin"


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



model = UNet2D(3, 1)
model = torch.load(model_save_path)
model = model.to(device)
model = model.eval()


m = nn.Sigmoid()
stats_list = []
count = 0
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
    
    frame_one = np.transpose(frame, (2, 0, 1))

    frame = frame_one[np.newaxis, :]
    label = label[np.newaxis, :]

    torch_frame_one = torch.from_numpy(frame).clone().float()
    torch_label_one = torch.from_numpy(label).clone().float()

    print(f"count: {count}================")
    data = torch_frame_one.to(device)
    label = torch_label_one.to(device)
    with torch.no_grad():
        output = model(data)
        output_sigmoid = m(output)
        output_cpu = output_sigmoid.to('cpu').detach().numpy().copy()
        label_cpu = label.to('cpu').detach().numpy().copy()

    label_cpu = np.where(label_cpu < 0.5, 0, 255)
    labels = label_cpu.astype(np.uint8)

    res = np.where(output_cpu < 0.5, 0, 1)
    res_inverse = np.where(res < 0.5, 1, 0)
    # tmp = res[0]
    # tmp = tmp.astype(np.uint8)
    masked = res[0] * frame[0]
    masked_inverse = res_inverse[0] * frame[0]

    res_transpose = np.transpose(res[0], (1,2,0))
    frame_transpose = np.transpose(frame[0], (1,2,0))
    masked_transpose = np.transpose(masked, (1,2,0))
    masked_inverse_transpose = np.transpose(masked_inverse, (1,2,0))

    # plt.imsave(f'{res_folder}/res_{count}.png', tmp[0], cmap='gray')
    # plt.imsave(f'{res_folder}/label_{count}.png', labels[0, :, :], cmap='gray')
    res_cv = cv.cvtColor((res_transpose*255).astype(np.float32), cv.COLOR_GRAY2BGR)
    frame = cv.cvtColor((frame_transpose*255).astype(np.float32), cv.COLOR_RGB2BGR)
    masked_bgr = cv.cvtColor((masked_transpose*255).astype(np.float32), cv.COLOR_RGB2BGR)
    masked_inverse_bgr = cv.cvtColor((masked_inverse_transpose*255).astype(np.float32), cv.COLOR_RGB2BGR)
    cv.imwrite(f"{res_folder}/res_{count}.png", res_cv) 
    cv.imwrite(f"{res_folder}/frame_{count}.png", frame) 
    cv.imwrite(f"{res_folder}/masked_{count}.png", masked_bgr) 
    cv.imwrite(f"{res_folder}/masked_inverse_{count}.png", masked_inverse_bgr) 
    img_merged = cv.addWeighted(frame, 0.8, res_cv, 0.2, 0.0, dtype=cv.CV_32F)
    img_merged_resize = cv.resize(img_merged, dsize=(640, 480))
    masked_bgr_resize = cv.resize(masked_bgr, dsize=(640, 480))
    masked_inverse_bgr_resize = cv.resize(masked_inverse_bgr, dsize=(640, 480))
    video.write(np.uint8(img_merged_resize))
    video_masked.write(np.uint8(masked_bgr_resize))
    video_masked_inverse.write(np.uint8(masked_inverse_bgr_resize))

    stats_list.append(calculate_stat_helper(output_cpu, label_cpu))

    df = pd.DataFrame(stats_list, columns=['precision', 'recall', 'fvalue', 'iou'])
    df.to_csv(f"{res_folder}/res.csv")
    count += 1



# def main(model_save_path, res_folder="res"):
#     bs = 1

#     device = torch.device('cuda')

#     #root_data = "/Users/kmihara/Downloads/video/*.mp4"
#     #root_label = "/Users/kmihara/Downloads/video_label/*.mp4"

#     root_data = "/home/ubuntu/workdir/nur/semantic-segmentation/data/CAR/training/video/*"
#     root_label = "/home/ubuntu/workdir/nur/semantic-segmentation/data/CAR/training/video_label/*"

#     paths_data = sorted(glob.glob(root_data))
#     paths_label = sorted(glob.glob(root_label))
#     print("list of sequentially data is:  ", paths_data)
#     print("list of sequentially label is: ", paths_label)


#     video_obj_data = {}
#     video_obj_label = {}
#     train_frames = {}
#     val_frames = {}
#     test_frames = {}

#     for data, label in zip(paths_data, paths_label):
#         video_data = cv.VideoCapture(data)
#         video_label = cv.VideoCapture(label)
#         video_obj_data[data]=video_data
#         video_obj_label[label]=video_label
#         train_frames[data] = [0, video_label.get(cv.CAP_PROP_FRAME_COUNT) // 2]
#         val_frames[data] = [video_label.get(cv.CAP_PROP_FRAME_COUNT) // 2,
#                                 video_label.get(cv.CAP_PROP_FRAME_COUNT) // 4 * 3]
#         test_frames[data] = [video_label.get(cv.CAP_PROP_FRAME_COUNT) // 4 * 3,
#                                     video_label.get(cv.CAP_PROP_FRAME_COUNT)]


#     valset = CarEval(paths_data, paths_label, video_obj_data, video_obj_label, train_frames)

    
#     model = UNet2D(3, 1)
#     model = torch.load(model_save_path)
#     model = model.to(device)
#     model = model.eval()

#     valloader = DataLoader(valset, batch_size=bs, num_workers=0, pin_memory=True)

#     m = nn.Sigmoid()

#     stats_list = []

#     count = 0

#     for i, (data, label) in enumerate(valloader):
#         print(f"count: {count}================")
#         data = data.to(device)
#         label = label.to(device)
#         with torch.no_grad():
#             output = model(data)
#             output_sigmoid = m(output)
#             output_cpu = output_sigmoid.to('cpu').detach().numpy().copy()
#             label_cpu = label.to('cpu').detach().numpy().copy()

#         label_cpu = np.where(label_cpu < 0.5, 0, 255)
#         labels = label_cpu.astype(np.uint8)

#         res = np.where(output_cpu < 0.5, 0, 255)
#         tmp = res[0]
#         tmp = tmp.astype(np.uint8)

#         plt.imsave(f'{res_folder}/res_{i}.png', tmp[0], cmap='gray')
#         plt.imsave(f'{res_folder}/label_{i}.png', labels[0, 0, :, :], cmap='gray')

#         stats_list.append(calculate_stat_helper(output_cpu, label_cpu))

#         df = pd.DataFrame(stats_list, columns=['precision', 'recall', 'fvalue', 'iou'])
#         df.to_csv(f"{res_folder}/res.csv")
#         count += 1


if __name__ == "__main__":
    print("hello, world")
    #main("tmp.pt")
