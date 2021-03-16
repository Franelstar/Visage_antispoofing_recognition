import cv2 as cv
import numpy as np
from skimage import feature as skif


# Calcul de l'histograme LBP
def lbp_histogram(image, P=8, R=1, method='nri_uniform'):
    lbp = skif.local_binary_pattern(image, P, R, method)  # lbp.shape is equal image.shape
    # cv2.imwrite("lbp.png",lbp)
    # plt.imshow(lbp)
    max_bins = int(lbp.max() + 1)  # max_bins is related P
    hist, _ = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))
    return lbp, hist


# Extraction de LBP
def extract_lbp_test(im):
    image_y = cv.cvtColor(im, cv.COLOR_BGR2YCrCb)
    lbp_y, y_h = lbp_histogram(image_y[:, :, 0])  # y channel
    lbp_cb, cb_h = lbp_histogram(image_y[:, :, 1])  # cb channel
    lbp_cr, cr_h = lbp_histogram(image_y[:, :, 2])  # cr channel
    feature_lbp = np.concatenate((y_h, cb_h, cr_h))
    return feature_lbp
