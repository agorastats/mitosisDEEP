import os
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import cv2
from tensorflow import keras
import segmentation_models as sm
from utils.keras.lossAndMetrics import bce_dice_loss, dice_coef


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
