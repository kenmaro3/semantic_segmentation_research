import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

class BottleNeck(Model):
    def __init__(self):
        super(BottleNeck, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(2048, activation="relu")
        self.dense2 = tf.keras.layers.Dense(512, activation="relu")
        self.reshape = tf.keras.layers.Reshape((8,8,8))
    def call(self,x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.reshape(x)
        return x

class DoubleConv(Model):
    def __init__(self, inc, outc, midc=None):
        super(DoubleConv, self).__init__()
        if midc is None:
            midc = outc

        self.conv1 = tf.keras.layers.Conv2D(filters=midc, kernel_size=(3, 3), strides=(1, 1), padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters=midc, kernel_size=(3, 3), strides=(1, 1), padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class Down(Model):
    # (maxpool for (h, w) not d=> DoubleConv)
    def __init__(self, inc, outc):
        super(Down, self).__init__()
        self.mp = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
        self.conv = DoubleConv(inc, outc)

    def call(self, x):
        # x.shape = (batch, d, h, w, c) # channel last
        x = self.mp(x)
        x = self.conv(x)
        return x


class Up(Model):
    # (ConvTranspose3D for (h, w) dimension, not d => DoubleConv)
    def __init__(self, inc, outc):
        super(Up, self).__init__()

        self.up = tf.keras.layers.Conv2DTranspose(filters=inc, kernel_size=(2, 2), strides=(2, 2))
        self.test = tf.keras.layers.Conv2D(inc,(1,1))
        self.conv = DoubleConv(inc, outc)


    def call(self, x1, x2):

        x1 = self.up(x1)
        # print("x1",x1.shape)
        # x2 = tf.keras.layers.Conv3D(x2.shape[4],(x2.shape[1],1,1))(x2)
        # print("x2 1",x2.shape)
        x2 = self.test(x2)
        # print("x2 2",x2.shape)
        x = tf.keras.layers.concatenate([x1,x2],axis=3)
        # x = self.concat([x1,x2])
        # x = self.up(x)
        x = self.conv(x)

        return x


class OutConv(Model):
    # last conv which maps to the same dimension of the target
    # use point wise convolution
    def __init__(self, inc, outc):
        super(OutConv, self).__init__()
        self.conv = tf.keras.layers.Conv2D(outc, kernel_size=(1, 1), strides=(1, 1), padding="same",activation="sigmoid")

    def call(self, x):
        # x.shape = (batch, d, h, w, inc)
        # self.conv(x).shape = (batch, d, h, w, outc)
        return self.conv(x)



    # def call(self,x):
    #     x = self.


class UNet2D(Model):
    def __init__(self, n_channels, n_classes):
        super(UNet2D, self).__init__()

        self.inc = DoubleConv(n_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.bottom = BottleNeck()

        self.up1 = Up(512, 512)
        # self.up2 = Up(512, 256,2)
        # self.up3 = Up(256, 128,4)
        # self.up4 = Up(128, 64,8)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.outc = OutConv(64, n_classes)


    def call(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(x4.shape)
        bottom = self.bottom(x5)

        # print(f'x3.shape >>> {x3.shape}')
        x = self.up1(bottom, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        # print(f'x.shape >>> {x.shape}')
        x = self.up4(x, x1)
        # print(f'x.shape >>> {x.shape}')

        logits = self.outc(x)

        return logits




if __name__ == "__main__":
    model = UNet2D(3, 1)
    #x = tf.experimental.numpy.random.randn(8, 64, 64, 3)
    bs = [1, 2, 4]
    res = []
    for batch in bs:
        x = np.random.rand(batch, 128, 128, 3)

        t_list = []
        for i in range(100):
            t1 = time.time()
            output = model(x)
            t2 = time.time()
            t_list.append(t2-t1)

            #print(f'time >>> {t2 - t1}')
        #print(output.shape)

        #print(model.summary())
        print(np.average(t_list[10:]))
        res.append(np.average(t_list[10:]))
    
    print("\n================")
    print(res)