import torch 
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple
import cv2 as cv
import random
import numpy as np
import glob
from matplotlib import pyplot as plt


class CarEval(Dataset):
    def __init__(self, paths_data, paths_label, obj_data, obj_label, frames) -> None:
        super().__init__()
        self.paths_data = paths_data
        self.paths_label = paths_label
        self.obj_data = obj_data
        self.obj_label = obj_label
        self.frames = frames
        self.data_size = 100

        self.datas = np.zeros((self.data_size, 3, 128, 128))
        self.labels = np.zeros((self.data_size, 1, 128, 128))
        for i in range(self.data_size):
            video_index = random.randint(0, len(self.paths_data)-1)
            target_video = self.paths_data[video_index]
            target_label = self.paths_label[video_index]

            frame_one = np.zeros((3, 128, 128))
            label_one = np.zeros((1, 128, 128), dtype=np.int32)


            random_frame = random.random()*0.99
            start = int((self.frames[target_video][1]-self.frames[target_video][0]) \
                        * random_frame \
                        + self.frames[target_video][0])
            try:
                self.obj_data[target_video].set(cv.CAP_PROP_POS_FRAMES, start)
                self.obj_label[target_label].set(cv.CAP_PROP_POS_FRAMES, start)
                ret_, frame_label = self.obj_label[target_label].read()
            except:
                print("fail to get img1")
                raise


            frame_label = cv.cvtColor(frame_label, cv.COLOR_BGR2GRAY)
            ret_, frame_label = cv.threshold(frame_label, 50, 255, cv.THRESH_BINARY)
            label = np.where(frame_label < 50.0 , 0, 1)

            ret, frame = self.obj_data[target_video].read()
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # frame = np.array(frame,dtype=np.float32)
            frame = frame/255.
            frame = frame.astype(np.float32)
            frame = cv.resize(frame, dsize=(128, 128))

            frame_one = np.transpose(frame, (2, 0, 1))
            frame_label = cv.resize(frame_label, dsize=(128, 128))


            self.datas[i] = frame_one
            self.labels[i] = frame_label


    def __len__(self) -> int:
        return self.data_size
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        torch_frame_one = torch.from_numpy(self.datas[index]).clone().float()
        #torch_label_one = torch.from_numpy(self.labels[index]).clone().long()
        torch_label_one = torch.from_numpy(self.labels[index]).clone()
        #return torch_frame_one, torch_label_one.squeeze()
        return torch_frame_one, torch_label_one


    
class Car(Dataset):
    def __init__(self, paths_data, paths_label, obj_data, obj_label, frames) -> None:
        super().__init__()
        self.paths_data = paths_data
        self.paths_label = paths_label
        self.obj_data = obj_data
        self.obj_label = obj_label
        self.frames = frames
        self.data_size = 200

        self.datas = np.zeros((self.data_size, 3, 128, 128))
        self.labels = np.zeros((self.data_size, 1, 128, 128))
        for i in range(self.data_size):
            video_index = random.randint(0, len(self.paths_data)-1)
            target_video = self.paths_data[video_index]
            target_label = self.paths_label[video_index]

            frame_one = np.zeros((3, 128, 128))
            label_one = np.zeros((1, 128, 128), dtype=np.int32)


            random_frame = random.random()*0.99
            start = int((self.frames[target_video][1]-self.frames[target_video][0]) \
                        * random_frame \
                        + self.frames[target_video][0])
            try:
                self.obj_data[target_video].set(cv.CAP_PROP_POS_FRAMES, start)
                self.obj_label[target_label].set(cv.CAP_PROP_POS_FRAMES, start)
                ret_, frame_label = self.obj_label[target_label].read()
            except:
                print("fail to get img1")
                raise


            frame_label = cv.cvtColor(frame_label, cv.COLOR_BGR2GRAY)
            ret_, frame_label = cv.threshold(frame_label, 50, 255, cv.THRESH_BINARY)
            label = np.where(frame_label < 50.0 , 0, 1)

            ret, frame = self.obj_data[target_video].read()
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # frame = np.array(frame,dtype=np.float32)
            frame = frame/255.
            frame = frame.astype(np.float32)

            crop_x = np.random.randint(0, frame.shape[1]-128-1)
            crop_y = np.random.randint(0, frame.shape[0]-128-1)
            # print(f"crop_y: {crop_y}")
            # print(f"crop_x: {crop_x}")
            # original_frame = frame
            # cv.imwrite('original.png', original_frame)
            tmp_frame = frame[crop_y:crop_y+128, crop_x:crop_x+128, :]
            frame_one = np.transpose(tmp_frame, (2, 0, 1))
            tmp_label= label[crop_y:crop_y+128, crop_x:crop_x+128]
            frame_label = tmp_label

            self.datas[i] = frame_one
            self.labels[i] = frame_label


    def __len__(self) -> int:
        return self.data_size
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        torch_frame_one = torch.from_numpy(self.datas[index]).clone().float()
        #torch_label_one = torch.from_numpy(self.labels[index]).clone().long()
        torch_label_one = torch.from_numpy(self.labels[index]).clone()
        #return torch_frame_one, torch_label_one.squeeze()
        return torch_frame_one, torch_label_one



    # def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
    #     print("debug0")
    #     video_index = random.randint(0, len(self.paths_data)-1)
    #     target_video = self.paths_data[video_index]
    #     target_label = self.paths_label[video_index]

    #     frame_one = np.zeros((3, 128, 128))
    #     label_one = np.zeros((1, 128, 128), dtype=np.int32)


    #     random_frame = random.random()*0.99
    #     start = int((self.frames[target_video][1]-self.frames[target_video][0]) \
    #                 * random_frame \
    #                 + self.frames[target_video][0])
    #     try:
    #         self.obj_data[target_video].set(cv.CAP_PROP_POS_FRAMES, start)
    #         self.obj_label[target_label].set(cv.CAP_PROP_POS_FRAMES, start)
    #         ret_, frame_label = self.obj_label[target_label].read()
    #     except:
    #         print("fail to get img1")
    #         raise


    #     frame_label = cv.cvtColor(frame_label, cv.COLOR_BGR2GRAY)
    #     ret_, frame_label = cv.threshold(frame_label, 50, 255, cv.THRESH_BINARY)
    #     label = np.where(frame_label < 50.0 , 0, 1)

    #     ret, frame = self.obj_data[target_video].read()
    #     frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    #     # frame = np.array(frame,dtype=np.float32)
    #     frame = frame/255.
    #     frame = frame.astype(np.float32)

    #     crop_x = np.random.randint(0, frame.shape[1]-128-1)
    #     crop_y = np.random.randint(0, frame.shape[0]-128-1)
    #     # print(f"crop_y: {crop_y}")
    #     # print(f"crop_x: {crop_x}")
    #     # original_frame = frame
    #     # cv.imwrite('original.png', original_frame)
    #     tmp_frame = frame[crop_y:crop_y+128, crop_x:crop_x+128, :]
    #     frame_one = np.transpose(tmp_frame, (2, 0, 1))
    #     tmp_label= label[crop_y:crop_y+128, crop_x:crop_x+128]
    #     frame_label = tmp_label


    #     torch_frame_one = torch.from_numpy(frame_one).clone().float()
    #     torch_label_one = torch.from_numpy(frame_label).clone().long()
    #     return torch_frame_one, torch_label_one.squeeze()



if __name__ == '__main__':
    # root_data = "/home/ubuntu/workdir/TargetDataSet-forModel/Location/*data*"
    root_data = "/home/ubuntu/workdir/nur/semantic-segmentation/data/CAR/training/video/*"
    # root_label = "/home/ubuntu/workdir/TargetDataSet-forModel/Location/*label*"*
    root_label = "/home/ubuntu/workdir/nur/semantic-segmentation/data/CAR/training/video_label/*"

    root_data = "/Users/kmihara/Downloads/video/*.mp4"
    root_label = "/Users/kmihara/Downloads/video_label/*.mp4"

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
        print(video_data.get(cv.CAP_PROP_FRAME_COUNT),video_label.get(cv.CAP_PROP_FRAME_COUNT))
        train_frames[data] = [0, video_label.get(cv.CAP_PROP_FRAME_COUNT) // 2]
        val_frames[data] = [video_label.get(cv.CAP_PROP_FRAME_COUNT) // 2,
                                video_label.get(cv.CAP_PROP_FRAME_COUNT) // 4 * 3]
        test_frames[data] = [video_label.get(cv.CAP_PROP_FRAME_COUNT) // 4 * 3,
                                    video_label.get(cv.CAP_PROP_FRAME_COUNT)]


    dataset = Car(paths_data,paths_label,video_obj_data,video_obj_label,train_frames)
    
    print(f"len: {len(dataset)}")
    test = dataset[0]
    print(test[0].size())
    print(test[1].size())
