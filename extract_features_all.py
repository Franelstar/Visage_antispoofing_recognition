from skimage import io
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import Model
import tensorflow as tf
import cv2 as cv
from skimage import feature as skif

model_loaded = tf.keras.models.load_model('models/feature_extraction.h5')
layer_name = 'dense_6'
intermediate_layer_model = Model(inputs=model_loaded.input, outputs=model_loaded.get_layer(layer_name).output)

def lbp_histogram(image, P=8, R=1, method='nri_uniform'):
    lbp = skif.local_binary_pattern(image, P, R, method)  # lbp.shape is equal image.shape
    # cv2.imwrite("lbp.png",lbp)
    # plt.imshow(lbp)
    max_bins = int(lbp.max() + 1)  # max_bins is related P
    hist, _ = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))
    return lbp, hist


def extract_lbp_test(im):
  image_y = cv.cvtColor(im, cv.COLOR_BGR2YCrCb)
  lbp_y, y_h = lbp_histogram(image_y[:,:,0]) # y channel
  lbp_cb, cb_h = lbp_histogram(image_y[:,:,1]) # cb channel
  lbp_cr, cr_h = lbp_histogram(image_y[:,:,2]) # cr channel
  feature_lbp = np.concatenate((y_h, cb_h, cr_h))
  return feature_lbp


def extract_tf_test(im):
  img = tf.image.resize(im, [224, 224], method='nearest')
  X = image.img_to_array(img)
  X = np.expand_dims(X, axis=0)
  images = np.vstack([X])
  images = images / 255
  feature_tensorflow = intermediate_layer_model.predict([images])[0]
  return feature_tensorflow