import logging
import os
import sys

import cv2
import numpy as np
import pandas as pd

from evaluation.base import EvaluationMasks, INFO_CSV_KEY, get_dice_coef
from evaluation.evaluationUOC import EvaluationMasksUOC
from utils.loadAndSaveResults import read_data_frame, store_data_frame
from utils.image import trace_boundingBox, rle_decode, create_dir, read_mask
from utils.runnable import Main

DATA_PATH = 'data/icpr12/validation_data/icpr12_evaluation'
MASK_PATH = 'data/icpr12/validation_data/icpr12_evaluation/masks_centroid' # eval with centroids and not shape masks


class EvaluationICPR12(EvaluationMasks):

    def __init__(self):
        super().__init__()
        self.data_path = DATA_PATH
        self.mask_path = MASK_PATH
        self.suffix_annot = ''
        # self.output_path = None

    def get_pred_info(self, options):
        infoDF = super().get_pred_info(options)
        # annotations have same name but different extension, change it
        infoDF.loc[:, 'id'] = infoDF.loc[:, 'id'].str.replace('bmp', 'jpg')
        return infoDF


if __name__ == '__main__':
    argv = sys.argv[1:]
    argv += ['--infoCsv',
             'icrp12_all_data_2022_03_20_resnet_custom_loss_alpha_05_beta_1_gamma_1.csv',
             '--output',
             'icrp12_all_data_2022_03_20_resnet_custom_loss_alpha_05_beta_1_gamma_1']

    Main(EvaluationICPR12()).run(argv)


    # all_data_2022_03_20_unet_custom_loss.csv: recall 0.783, prec 0.365, fscore: 0.467
    # icpr12_pred_info_all_data_2022_03_20_unet.csv: recall 0.836, prec: 0.24, fscore: 0.348
    # icpr12_pred_info_all_data_2022_03_20_resnet.csv: recall 0.72, prec: 0.38, fscore: 0.47
    # all_data_2022_03_20_unet_custom_loss_gamma3.csv: 