import logging
import os
import cv2
import numpy as np
import pandas as pd

from abc import ABCMeta

from patchify import patchify

from utils.image import create_dir, read_mask, rle_decode, trace_boundingBox
from utils.loadAndSaveResults import read_data_frame, store_data_frame
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
        infoDF = self.get_pred_info(options)
        self.iterate_over_pred_images(infoDF, options)

    def get_image_name(self, img):
        image_name = img['id'].split('.')[0] + self.suffix_annot + '.' + img['id'].split('.')[1]
        return image_name
    def iterate_over_pred_images(self, infoDF, options):
        dice_coef_df_list = [pd.DataFrame()]
        for i, img in infoDF.iterrows():
            logging.info('__evaluation image: %s' % str(img['id']))
            image_name = self.get_image_name(img)
            image = cv2.imread(os.path.join(self.data_path, image_name))
            mask = read_mask(os.path.join(self.mask_path, image_name))
            stain = False if img.get('stain') is None else img.get('stain')
            dice_coef_df = pd.DataFrame({'img': img['id'], 'stain': stain, 'dice_all': 0,
                                         'pred_mitotic': 0, 'pred_mitotic_min_shape': 0,
                                         'similar_centroid': 0, 'dice_patch': 0, 'recall': 0}, index=[0])

            mask = cv2.resize(mask, (img['size_x'], img['size_y']))
            dice_coef_df.loc[:, 'obs_mitotic'] = len(get_centroids_of_mask(mask))
            image = cv2.resize(image, (img['size_x'], img['size_y']))
            if len(img['rle']) > 0:
                pred_mask = rle_decode(img['rle'], np.array((img['size_y'], img['size_x'])))
                # smooth image using dilate and erode methods
                kernel5 = np.ones((10, 10), np.uint8)  # to use in cv2 methods
                pred_mask = cv2.dilate(pred_mask, kernel5, iterations=1)
                pred_mask = cv2.erode(pred_mask, kernel5, iterations=1)
                pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel5)
                dice_coef_df.loc[:, 'pred_mitotic'] = len(get_centroids_of_mask(pred_mask))
                dice_coef_df.loc[:, 'pred_mitotic_min_shape'] = len(get_centroids_of_mask(pred_mask, min_shape=15))
                image = trace_boundingBox(image, pred_mask, color=(0, 0, 0), width=15)
                dice_coef = get_dice_coef((mask / 255.).astype(np.uint8), pred_mask)
                dice_coef_df.loc[:, 'dice_all'] = dice_coef
                size_x = (mask.shape[1] // 256) * 256
                size_y = (mask.shape[0] // 256) * 256
                mask = cv2.resize(mask, (size_x, size_y))
                pred_mask = cv2.resize(pred_mask, (size_x, size_y))
                patches_mask = patchify(mask, (256, 256), step=256)  # Step=256 for 256 patches means no overlap
                patches_pred_mask = patchify(pred_mask, (256, 256), step=256)
                dice_coef_list = list()
                for i in range(patches_mask.shape[0]):
                    for j in range(patches_mask.shape[1]):
                        single_patch_mask = patches_mask[i, j, :, :]
                        single_patch_pred_mask = patches_pred_mask[i, j, :, :]
                        aux_dice = get_dice_coef((single_patch_mask / 255.).astype(np.uint8), single_patch_pred_mask)
                        dice_coef_list.append(aux_dice)
                dice_coef_df.loc[:, 'dice_patch'] = pd.Series(dice_coef_list).mean()
                dice_coef_df.loc[:, 'recall'] = float(get_recall((mask / 255.), pred_mask))
                dice_coef_df.loc[:, 'similar_centroid'] = compare_centroids_of_masks(mask, pred_mask)

            cv2.imwrite(os.path.join(self.output_path, 'stain' + str(stain) + '_' + image_name), image)
            dice_coef_df_list.append(dice_coef_df)
        dice_coef_df = pd.concat(dice_coef_df_list, ignore_index=True)
        dice_coef_df.rename(columns={'index': 'image', 0: 'dice_coef'}, inplace=True)
        # metrica definitiva
        dice_coef_df.loc[:, 'recall_bo'] = dice_coef_df.loc[:, 'similar_centroid'] / dice_coef_df.loc[:, 'obs_mitotic']
        dice_coef_df.loc[:, 'prec_bo'] = dice_coef_df.loc[:, 'similar_centroid'] / dice_coef_df.loc[:,
                                                                                   'pred_mitotic_min_shape']
        dice_coef_df.loc[:, 'fscore'] = (2 * dice_coef_df.loc[:, 'recall_bo'] * dice_coef_df.loc[:, 'prec_bo']) / (
                dice_coef_df.loc[:, 'recall_bo'] + dice_coef_df.loc[:, 'prec_bo'])
        store_data_frame(dice_coef_df, os.path.join(self.output_path, options[OUTPUT_PATH_KEY] + '_dice_coef.csv'))
