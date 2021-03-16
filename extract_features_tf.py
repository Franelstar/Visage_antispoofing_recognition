from skimage import io
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import Model
import tensorflow as tf
import cv2 as cv

model_loaded = tf.keras.models.load_model('models/feature_extraction.h5')
layer_name = 'dense_6'
intermediate_layer_model = Model(inputs=model_loaded.input, outputs=model_loaded.get_layer(layer_name).output)

def extract_tf_test(im):
  img = tf.image.resize(im, [224, 224], method='nearest')
  X = image.img_to_array(img)
  X = np.expand_dims(X, axis=0)
  images = np.vstack([X])
  images = images / 255
  feature_tensorflow = intermediate_layer_model.predict([images])[0]
  return feature_tensorflow