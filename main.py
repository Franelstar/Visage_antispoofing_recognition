import streamlit as st
import cv2 as cv
import os
import numpy as np
from extract_features_lbp import extract_lbp_test
from pca_reduction import pca_reduction
from predict import predict
from extract_features_tf import extract_tf_test

feature_type = st.sidebar.radio("Type de features", ('Only LBP', 'Only Tensorflow', 'LBP + Tensorflow'))
model = st.sidebar.radio("Model", ('SVM', 'Logistic regression', 'Random Forest', 'Decision Tree', 'SGD'))

uploaded_files = st.file_uploader("Choisissez les images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
list_images = []
predire = {0: "Live", 1: "Spoof"}

if uploaded_files is not None:
    list_images = [cv.imread('test/'+uploaded_file.name) for uploaded_file in uploaded_files]

    if list_images:

        images_test_ = []
        prediction = []

        if feature_type == 'Only LBP':
            features_lbp = [extract_lbp_test(im) for im in list_images]

            images_test_ = pca_reduction('lbf_pca.sav', features_lbp)
            if model == 'SVM':
                prediction = predict('lbf_svm.sav', images_test_)
            elif model == 'Logistic regression':
                prediction = predict('lbf_lr.sav', images_test_)
            elif model == 'Random Forest':
                prediction = predict('lbf_rf.sav', images_test_)
            elif model == 'Decision Tree':
                prediction = predict('lbp_dt.sav', images_test_)
            else:
                prediction = predict('lbf_sgd.sav', images_test_)

        elif feature_type == 'Only Tensorflow':
            features_tf = [extract_tf_test(im) for im in list_images]

            images_test_ = pca_reduction('tf_pca.sav', features_tf)
            if model == 'SVM':
                prediction = predict('tf_svm.sav', images_test_)
            elif model == 'Logistic regression':
                prediction = predict('tf_lr.sav', images_test_)
            elif model == 'Random Forest':
                prediction = predict('tf_rf.sav', images_test_)
            elif model == 'Decision Tree':
                prediction = predict('tf_dt.sav', images_test_)
            else:
                prediction = predict('tf_sgd.sav', images_test_)

        else:
            features_lbp = [extract_lbp_test(im) for im in list_images]
            features_tf = [extract_tf_test(im) for im in list_images]
            features_all = [np.concatenate((f_lbp, f_tf), axis=None) for f_lbp, f_tf in zip(features_lbp, features_tf)]

            images_test_ = pca_reduction('all_pca.sav', features_all)
            if model == 'SVM':
                prediction = predict('all_svm.sav', images_test_)
            elif model == 'Logistic regression':
                prediction = predict('all_lr.sav', images_test_)
            elif model == 'Random Forest':
                prediction = predict('all_rf.sav', images_test_)
            elif model == 'Decision Tree':
                prediction = predict('all_dt.sav', images_test_)
            else:
                prediction = predict('all_sgd.sav', images_test_)

        list_images = [cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150)) for img in list_images]

        for i, im in enumerate(images_test_):
            col1, col2 = st.beta_columns(2)
            col1.image(list_images[i])
            col2.write('Prédiction : {}'.format(predire[prediction[i]]))