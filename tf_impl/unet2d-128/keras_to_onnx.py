import tensorflow as tf
from tensorflow import keras
#from unet2d_mobile_ultra_thin import UNet2D
from unet2d_mobile_5m import UNet2D


## Generate model
inputs  = tf.keras.layers.Input(shape=(128, 128, 3))
model   = UNet2D(n_channels = 3, n_classes = 1)        ##Note: Do we need to simplify this too?
outputs = model(inputs)
model  = tf.keras.Model(inputs, outputs)

model.load_weights(f"model_128_5m.h5")

tf.saved_model.save(model, "tmp_model_5m")

# then
# python -m tf2onnx.convert --saved-model tmp_model_5m --output "model_128_5m.onnx"
