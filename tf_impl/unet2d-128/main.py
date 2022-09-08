
import cv2 as cv
import glob
# from unet_network import UNet3D
from unet2d import UNet2D
import tensorflow as tf
from dl_train import DataLoader
from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


# # Variables-------------------
# data_rand_num = 8
# label_rand_num = 1
# x_crop_size = 64
# y_crop_size = 64
# data_rgb = 3
# label_rgb = 1
# height_size = 480
# width_size = 640

l_rate = 0.001
beta1 = 0.9
beta2 = 0.999
eps = 1e-07
momen = 0.1
nesterov = True


def rotate(l, n):
    return l[-n:] + l[:-n]


def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = tf.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def weighted_binary_crossentropy(y_true, y_pred):
    loss = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, 3)
    return loss



# root_data = "/home/ubuntu/workdir/TargetDataSet-forModel/Location/*data*"
#root_data = "/home/ubuntu/workdir/nur/semantic-segmentation/data/CAR/training/video/*"
# root_label = "/home/ubuntu/workdir/TargetDataSet-forModel/Location/*label*"*
#root_label = "/home/ubuntu/workdir/nur/semantic-segmentation/data/CAR/training/video_label/*"
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
train_gen = DataLoader(paths_data,paths_label,video_obj_data,video_obj_label,train_frames, batch_size=4)
val_gen = DataLoader(paths_data,paths_label,video_obj_data,video_obj_label,val_frames, batch_size=4)


## Generate the model
inputs  = tf.keras.layers.Input(shape=(128,128, 3))
model   = UNet2D(n_channels=3,n_classes=1)
outputs = model(inputs)
model   = tf.keras.Model(inputs,outputs)

## Summarizing the model
model.summary()

## Set optimizer
ADAM = tf.keras.optimizers.Adam(learning_rate=l_rate,
                                beta_1=beta1,
                                beta_2=beta2,
                                epsilon=eps)  # DEFAULT Optimizer
SGD = tf.keras.optimizers.SGD(learning_rate=l_rate,
                              momentum=momen,
                              nesterov=nesterov)
NADAM = tf.keras.optimizers.Nadam(learning_rate=l_rate,
                                  beta_1=beta1,
                                  beta_2=beta2,
                                  epsilon=eps)


## Set rule for stopping
checkpoint = ModelCheckpoint("model_128.h5",
                             verbose=1,
                             save_freq='epoch',
                             save_best_only=True,
                             save_weights_only=True)
early = EarlyStopping(monitor='val_loss',
                      min_delta=0,
                      patience=6,
                      verbose=1,
                      mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              patience=3,
                              factor=0.1,
                              min_lr=1e-7)

## Model compilation
model.compile(loss= bce_dice_loss,
              optimizer=ADAM,
              metrics=['acc'])

## Fitting the model
r= model.fit(train_gen,
             validation_data=val_gen,
             epochs=100,
             steps_per_epoch=300,
             validation_steps=100,
             callbacks=[checkpoint, early, reduce_lr])

## Visualizing the loss function
fig = plt.figure()
plt.plot(r.history['loss'], label='loss')
plt.title('Model loss-ADAM')
plt.xlabel('Epoch')
plt.ylabel('Loss function')
plt.legend()
plt.show()
fig.savefig("model_128.jpg", transparent=True)
