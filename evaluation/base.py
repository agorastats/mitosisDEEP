import logging
import os
import cv2
import numpy as np
import pandas as pd

from abc import ABCMeta
from utils.image import create_dir
from utils.loadAndSaveResults import read_data_frame
from utils.runnable import Runnable
from keras import backend as K

# ICPR12: A segmented mitosis would be counted as correctly detected if its centroid
# is localised within a range of 8 μm from the centroid of a ground truth mitosis.
# aprox 0.25um per pixel in images ICPR12, so aprox. 8/0.25 as tolerance
TOLERANCE_PIXEL_ERROR = 30


def get_dice_coef(mask1, mask2):
    intersect = np.sum(mask1 * mask2)
    fsum = np.sum(mask1)
    ssum = np.sum(mask2)
    dice = (2 * intersect + 1) / (fsum + ssum + 1)
    dice = np.mean(dice).round(3)
    return dice

def get_recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def get_centroids_of_mask(mask, min_shape=0):
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    centroidDF = pd.DataFrame([c.mean(axis=0).astype(int).tolist()[0] for c in contours if c.shape[0] >= min_shape])
    return centroidDF


def compare_centroids_of_masks(mask, pred_mask, tol=TOLERANCE_PIXEL_ERROR):
    obsDF = get_centroids_of_mask(mask)
    predDF = get_centroids_of_mask(pred_mask, min_shape=15)  # not consider less than this shape
    # compare it
    match = 0
    for i, row in predDF.iterrows():
        if obsDF.empty:
            return match
        match_cond = (np.abs(row - obsDF) <= tol).all(axis=1)
        if match_cond.any():
            # drop and count match
            obsDF = obsDF.loc[~match_cond, :]
            match += 1
    return match


INFO_CSV_KEY = 'info_csv'
OUTPUT_PATH_KEY = 'outputPath'


class EvaluationMasks(Runnable, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()
        self.data_path = None
        self.mask_path = None
        self.pred_info = None
        self.images_list = None
        self.output_path = None
        self.suffix_annot = ''

    def add_options(self, parser):
        super().add_options(parser)
        parser.add_option('--infoCsv', dest=INFO_CSV_KEY,
                          help='Folder containing rle encode prediction masks info (csv format)',
                          action='store', default=None)
        parser.add_option('--output', dest=OUTPUT_PATH_KEY,
                          help='Folder to output annotation images with mask predictions',
                          action='store', default=None)

    def create_needed_folders(self, options):
        info_csv = options[INFO_CSV_KEY]
        info_csv_name = info_csv.split('.')[0]
        self.output_path = os.path.join(self.data_path, info_csv_name) if options[OUTPUT_PATH_KEY] is None \
            else os.path.join(self.data_path, options[OUTPUT_PATH_KEY])
        create_dir(self.output_path)

    def get_pred_info(self, options):
        infoDF = read_data_frame(os.path.join(self.data_path, options[INFO_CSV_KEY]))
        assert not infoDF.empty, 'infoDF of predictions are empty!'
        infoDF.loc[:, 'rle'].fillna('', inplace=True)
        return infoDF

    def pre_run(self, options):
        assert options[INFO_CSV_KEY] is not None, 'need to inform pred_info file'
        self.create_needed_folders(options)
        logging.info('iterating process: %s' % self.__class__.__name__)

    def run(self, options):
        pass
