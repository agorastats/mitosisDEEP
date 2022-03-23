import logging
import os
import sys

import cv2
import numpy as np
import pandas as pd

from evaluation.base import EvaluationMasks, get_dice_coef, get_centroids_of_mask, \
    compare_centroids_of_masks
from utils.loadAndSaveResults import store_data_frame
from utils.image import trace_boundingBox, rle_decode, read_mask

# OUTPUT_PATH = 'data/Laura/pred_proves_dataset20220207_unet_bce_dice_loss_stain'
# PRED_INFO_CSV = 'pred_proves_dataset20220207_unet_bce_dice_loss_stain.csv'
from utils.runnable import Main

DATA_PATH = 'data/UOC Mitosis'
MASK_PATH = 'data/UOC Mitosis/masks'


# --infoCsv pred_proves_dataset20220207_unet_bce_dice_loss_stain.csv'

class EvaluationMasksUOC(EvaluationMasks):

    def __init__(self):
        super().__init__()
        self.data_path = DATA_PATH
        self.mask_path = MASK_PATH
        self.suffix_annot = 'x'
        # self.output_path = None

    def run(self, options):
        infoDF = self.get_pred_info(options)
        dice_coef_df_list = [pd.DataFrame(columns=['img', 'stain', 'dice'])]
        for i, img in infoDF.iterrows():
            logging.info('__evaluation image: %s' % str(img['id']))
            image_name = img['id'].split('.')[0] + self.suffix_annot + '.' + img['id'].split('.')[1]
            image = cv2.imread(os.path.join(self.data_path, image_name))
            mask = read_mask(os.path.join(self.mask_path, image_name))
            stain = False if img.get('stain') is None else img.get('stain')
            dice_coef_df = pd.DataFrame({'img': img['id'], 'stain': stain, 'dice': np.nan,
                                         'pred_mitotic': 0, 'pred_mitotic_min_shape': 0,
                                         'similar_centroid': 0}, index=[0])

            mask = cv2.resize(mask, (img['size_x'], img['size_y']))
            dice_coef_df.loc[:, 'obs_mitotic'] = len(get_centroids_of_mask(mask))
            image = cv2.resize(image, (img['size_x'], img['size_y']))
            if len(img['rle']) > 0:
                pred_mask = rle_decode(img['rle'], np.array((img['size_y'], img['size_x'])))
                dice_coef_df.loc[:, 'pred_mitotic'] = len(get_centroids_of_mask(pred_mask))
                dice_coef_df.loc[:, 'pred_mitotic_min_shape'] = len(get_centroids_of_mask(pred_mask, min_shape=15))
                image = trace_boundingBox(image, pred_mask, color=(0, 0, 0), width=15)
                dice_coef = get_dice_coef((mask / 255.).astype(np.uint8), pred_mask)
                dice_coef_df.loc[:, 'dice'] = dice_coef
                dice_coef_df.loc[:, 'similar_centroid'] = compare_centroids_of_masks(mask, pred_mask)

            cv2.imwrite(os.path.join(self.output_path, 'stain' + str(stain) + '_' + image_name), image)
            dice_coef_df_list.append(dice_coef_df)
        dice_coef_df = pd.concat(dice_coef_df_list, ignore_index=True)
        dice_coef_df.rename(columns={'index': 'image', 0: 'dice_coef'}, inplace=True)
        store_data_frame(dice_coef_df, os.path.join(self.output_path, 'dice_coef.csv'))


if __name__ == '__main__':

    argv = sys.argv[1:]
    # argv += ['--infoCsv',
    #          'pred_proves_dataset20220207_unet_bce_dice_loss_stain.csv',
    #          '--output',
    #          'pred_proves_dataset20220207_unet_bce_dice_loss_stain']

    argv += ['--infoCsv',
             'pred_info_all_data_2022_03_20_unet.csv',
             '--output',
             'pred_info_all_data_2022_03_20_unet']

    Main(EvaluationMasksUOC()).run(argv)
