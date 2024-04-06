import os
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import cv2
import segmentation_models as sm
from tensorflow import keras
from utils.keras.lossAndMetrics import bce_dice_loss, dice_coef
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers import Lambda, RepeatVector, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, GlobalMaxPool2D
from keras.layers import concatenate, add

# PRETRAINED BACKBONE MODEL
def get_backbone_unet(backbone='resnet34', compile_it=True, weights=None):
    # avoid error set framework keras: https://stackoverflow.com/questions/67792138/attributeerror-module-keras-utils-has-no-attribute-get-file-using-segmentat
    sm.set_framework('tf.keras')
    sm.framework()
    # BUILD UNET WITH ENCODER BACKBONE (IMAGENET)
    model = sm.Unet(backbone, input_shape=(256, 256, 3), classes=1, activation='sigmoid')  # encoder_freeze=True
    if compile_it:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=bce_dice_loss, metrics=[dice_coef])

    if backbone == 'resnet34':
        # for resnet34 backbone sm method return unity: same value and we expected 255 (standard preprocess)
        preprocess_input = None
    else:
        preprocess_input = sm.get_preprocessing(backbone)

    if weights is not None:
        model.load_weights(weights)
    return model, preprocess_input


def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def build_unet(n_filters = 16, dropout = 0.05, batchnorm = True):
    """Function to define the UNET Model"""
    input_img = Input((256, 256, 3), name='img')  # fix input, for mitosis, patches are 256x256 and RGB
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

    # last layer
    outputModified = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    # create model
    model = Model(inputs=[input_img], outputs=[outputModified])
    return model

# to test it... for instance...
# model = build_unet(n_filters=16, dropout=0.5, batchnorm=True)
# model.load_weights("drive/MyDrive/mitosis_data/pretrained-model-monuseg-unet.h5", by_name=True)  # INIT WEIGHTS BY NAME ...

