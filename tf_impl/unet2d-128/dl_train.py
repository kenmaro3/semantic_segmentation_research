from multiprocessing.util import is_exiting
import cv2 as cv
import random
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
from pyparsing import original_text_for
from scipy.ndimage.interpolation import rotate
#from scipy.misc import imresize

import tensorflow as tf

def rotate(l, n):
    return l[-n:] + l[:-n]

class DataLoader(tf.keras.utils.Sequence):
    def __init__(self,
                 paths_data,
                 paths_label,
                 obj_data,
                 obj_label,
                 frames,
                 batch_size=16,
                 h_flip=False,
                 v_flip=False,
                 rotation=False,
                 is_gray_scale=False
                 ):
        self.paths_data = paths_data
        self.paths_label = paths_label
        self.obj_data = obj_data
        self.obj_label = obj_label
        self.frames = frames
        self.batch_size = batch_size
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.rotation = rotation
        self.is_gray_scale = is_gray_scale

    def _horizontal_flip(self, frame, label, rate=0.5):
        if np.random.rand() < rate:
            frame = frame[:, ::-1, :]
            label = label[:, ::-1]
        return frame, label

    def _vertical_flip(self, frame, label, rate=0.5):
        if np.random.rand() < rate:
            frame = frame[::-1, :, :]
            label = label[::-1, :]
        return frame, label

    def _random_rotation(self, frame, label, angle_range=(0, 180)):
        h, w, _ = frame.shape
        angle = np.random.randint(*angle_range)
        frame = rotate(frame, angle)
        frame = imresize(frame, (h, w))
        label = rotate(label, angle)
        label = imresize(label, (h, w))
        return frame, label


    def __getitem__(self, idx):
        target_video = self.paths_data[0]
        target_label = self.paths_label[0]

        if self.is_gray_scale:
            batch_frame = np.zeros((self.batch_size, 128, 128, 1))
        else:
            batch_frame = np.zeros((self.batch_size, 128, 128, 3))
        batch_label = np.zeros((self.batch_size, 128, 128))

        for i in range(self.batch_size):

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

            if self.is_gray_scale:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame = frame[:, :, np.newaxis]
            else:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # frame = np.array(frame,dtype=np.float32)
            frame = frame/255.
            frame = frame.astype(np.float32)

            if self.h_flip:
                frame, label = self._horizontal_flip(frame, label)
            if self.v_flip:
                frame, label = self._vertical_flip(frame, label)
            if self.rotation:
                frame, label = self._random_rotation(frame, label, (-20, 20))

            crop_x = np.random.randint(0, frame.shape[1]-128-1)
            crop_y = np.random.randint(0, frame.shape[0]-128-1)
            # print(f"crop_y: {crop_y}")
            # print(f"crop_x: {crop_x}")
            # original_frame = frame
            # cv.imwrite('original.png', original_frame)
            frame = frame[crop_y:crop_y+128, crop_x:crop_x+128, :]
            label = label[crop_y:crop_y+128, crop_x:crop_x+128]

            batch_frame[i] = frame
            batch_label[i] = label
        # cv.imwrite('crop.png', frame)

        self.paths_data = rotate(self.paths_data, 1)
        self.paths_label = rotate(self.paths_label, 1)

        frame = frame[np.newaxis, :]
        label = label[np.newaxis, :]

        if self.is_gray_scale:
            assert batch_frame.shape == (self.batch_size, 128, 128, 1), f"{batch_frame.shape}"
        else:
            assert batch_frame.shape == (self.batch_size, 128, 128, 3), f"{batch_frame.shape}"
        assert batch_label.shape == (self.batch_size, 128, 128), f"{batch_label.shape}"

        return batch_frame, batch_label
        

    def __len__(self):
        return 300

    def on_epoch_end(self):
        pass


def get_dl(is_gray_scale=False):
    root_data = "/home/ubuntu/workdir/nur/semantic-segmentation/data/CAR/training/video/*"
    # root_label = "/home/ubuntu/workdir/TargetDataSet-forModel/Location/*label*"*
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
    count = 0

    for data, label in zip(paths_data, paths_label):
        video_data = cv.VideoCapture(data)
        video_label = cv.VideoCapture(label)
        video_obj_data[data]=video_data
        video_obj_label[label]=video_label
        print(video_data.get(cv.CAP_PROP_FRAME_COUNT),video_label.get(cv.CAP_PROP_FRAME_COUNT))
        if count % 4 == 0:
            train_frames[data] = [0, video_label.get(cv.CAP_PROP_FRAME_COUNT) // 2]
            val_frames[data] = [video_label.get(cv.CAP_PROP_FRAME_COUNT) // 2,
                                    video_label.get(cv.CAP_PROP_FRAME_COUNT) // 4 * 3]
            test_frames[data] = [video_label.get(cv.CAP_PROP_FRAME_COUNT) // 4 * 3,
                                    video_label.get(cv.CAP_PROP_FRAME_COUNT)]
        elif count % 4 == 1:
            train_frames[data] = [0, video_label.get(cv.CAP_PROP_FRAME_COUNT) // 2]
            test_frames[data] = [video_label.get(cv.CAP_PROP_FRAME_COUNT) // 2,
                                    video_label.get(cv.CAP_PROP_FRAME_COUNT) // 4 * 3]
            val_frames[data] = [video_label.get(cv.CAP_PROP_FRAME_COUNT) // 4 * 3,
                                    video_label.get(cv.CAP_PROP_FRAME_COUNT)]
        elif count % 4 == 2:
            train_frames[data] = [video_label.get(cv.CAP_PROP_FRAME_COUNT) // 2 , video_label.get(cv.CAP_PROP_FRAME_COUNT)]
            test_frames[data] = [0 , video_label.get(cv.CAP_PROP_FRAME_COUNT)//4]
            val_frames[data] = [ video_label.get(cv.CAP_PROP_FRAME_COUNT)//4,video_label.get(cv.CAP_PROP_FRAME_COUNT)//2]
        elif count % 4 == 3:
            train_frames[data] = [video_label.get(cv.CAP_PROP_FRAME_COUNT) // 2, video_label.get(cv.CAP_PROP_FRAME_COUNT)]
            val_frames[data] = [0, video_label.get(cv.CAP_PROP_FRAME_COUNT) // 4]
            test_frames[data] = [video_label.get(cv.CAP_PROP_FRAME_COUNT) // 4, video_label.get(cv.CAP_PROP_FRAME_COUNT) // 2]

        count += 1


    ## Build the object data
    train_gen = DataLoader(paths_data,paths_label,video_obj_data,video_obj_label,train_frames, batch_size=4, is_gray_scale=is_gray_scale)

    return train_gen


if __name__ == "__main__":
    train_gen = get_dl(is_gray_scale=True)
    data, label = train_gen[0]
    print(f"data.shape: {data.shape}")
    print(f"label.shape: {label.shape}")
    quit()
    print("here1")
    data = data*255.0
    data = data.astype(np.uint8)
    pil_image=Image.fromarray(data[0])
    pil_image.save("pil.png")
    # pil_image=Image.fromarray(label[0])
    print("\n\n==============")
    print(label.shape)
    plt.imsave('label.png', label[0], cmap='gray')
    # pil_image.save("pil_label.png")
