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

DATA_PATH = 'data/Laura'
MASK_PATH = 'data/Laura/masks'


# --infoCsv pred_proves_dataset20220207_unet_bce_dice_loss_stain.csv'

class EvaluationMasksLaura(EvaluationMasksUOC):

    def __init__(self):
        super().__init__()
        self.data_path = DATA_PATH
        self.mask_path = MASK_PATH
        self.suffix_annot = 'a'
        # self.output_path = None

if __name__ == '__main__':
    argv = sys.argv[1:]
    argv += ['--infoCsv', 'pred_proves_dataset20220207_unet_bce_dice_loss_stain.csv']

    Main(EvaluationMasksLaura()).run(argv)
