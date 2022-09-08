import cv2 as cv
import random
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
from pyparsing import original_text_for

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
                 batch_size=16):
        self.paths_data = paths_data
        self.paths_label = paths_label
        self.obj_data = obj_data
        self.obj_label = obj_label
        self.frames = frames
        self.batch_size = batch_size

    def __getitem__(self, idx):
        target_video = self.paths_data[0]
        target_label = self.paths_label[0]

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
            frame_label = cv.resize(frame_label, dsize=(128, 128))
            label = np.where(frame_label < 50.0 , 0, 1)

            ret, frame = self.obj_data[target_video].read()
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame = cv.resize(frame, dsize=(128, 128))
            # frame = np.array(frame,dtype=np.float32)
            frame = frame/255.
            frame = frame.astype(np.float32)


            batch_frame[i] = frame
            batch_label[i] = label
        # cv.imwrite('crop.png', frame)

        self.paths_data = rotate(self.paths_data, 1)
        self.paths_label = rotate(self.paths_label, 1)

        frame = frame[np.newaxis, :]
        label = label[np.newaxis, :]


        assert batch_frame.shape == (self.batch_size, 128, 128, 3), f"{batch_frame.shape}"
        assert batch_label.shape == (self.batch_size, 128, 128), f"{batch_label.shape}"

        return batch_frame, batch_label
        

    def __len__(self):
        return 300

    def on_epoch_end(self):
        pass


def get_dl(batch_size):
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


    ## Build the object data
    train_gen = DataLoader(paths_data,paths_label,video_obj_data,video_obj_label,train_frames, batch_size=batch_size)

    return train_gen


if __name__ == "__main__":
    train_gen = get_dl(1)
    data, label = train_gen[0]
    print(f"data.shape: {data.shape}")
    print(f"label.shape: {label.shape}")
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

