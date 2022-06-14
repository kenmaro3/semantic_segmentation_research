import cv2 as cv
import numpy as np
import tensorflow as tf
from unet2d import UNet2D
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    inputs  = tf.keras.layers.Input(shape=(128, 128, 3))
    model   = UNet2D(n_channels = 3, n_classes = 1)        ##Note: Do we need to simplify this too?
    outputs = model(inputs)
    model  = tf.keras.Model(inputs, outputs)

    model.load_weights(f"model_128.h5")

    input_model = "saved_model"
    output_model = "model_128_dynamic.tflite"

    print("\n\nhere0000!!!=================")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    print("\n\nhere!!!=================")
    with open(output_model, 'wb') as o_:
        o_.write(tflite_model)



    # model.save(input_model)
    # converter = tf.lite.TFLiteConverter.from_saved_model(input_model)
    # tflite_quant_model = converter.convert()
    # with open(output_model, 'wb') as o_:
    #     o_.write(tflite_quant_model)
