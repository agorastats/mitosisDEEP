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

DATA_PATH = 'data/Paula'
MASK_PATH = 'data/Paula/masks'


class EvaluationMasksPaula(EvaluationMasksUOC):

    def __init__(self):
        super().__init__()
        self.data_path = DATA_PATH
        self.mask_path = MASK_PATH
        self.suffix_annot = 'x'
        # self.output_path = None


if __name__ == '__main__':
    # todo: a determinar
    argv = sys.argv[1:]
    argv += ['--infoCsv',
             'pred_info_all_data_2022_03_20_unet.csv',
             '--output',
             'all_data_2022_03_20_unet']

    Main(EvaluationMasksPaula()).run(argv)
