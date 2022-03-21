import logging
import os

import cv2
import numpy as np
import pandas as pd

from evaluation.base import EvaluationMasks, INFO_CSV_KEY, get_dice_coef
from utils.loadAndSaveResults import read_data_frame, store_data_frame
from utils.image import trace_boundingBox, rle_decode, create_dir, read_mask

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
        dice_coef_dict = dict()
        for i, img in infoDF.iterrows():
            logging.info('__evaluation image: %s' % str(img['id']))
            image_name = img['id'].split('.')[0] + self.suffix_annot + '.' + img['id'].split('.')[1]
            image = cv2.imread(os.path.join(self.data_path, image_name))
            mask = read_mask(os.path.join(self.mask_path, image_name))
            dice_coef = np.nan
            if len(img['rle']) > 0:
                mask = cv2.resize(mask, (img['size_x'], img['size_y']))
                image = cv2.resize(image, (img['size_x'], img['size_y']))
                pred_mask = rle_decode(img['rle'], np.array((img['size_y'], img['size_x'])))
                image = trace_boundingBox(image, pred_mask, color=(0, 0, 0), width=15)
                dice_coef = get_dice_coef((mask / 255.).astype(np.uint8), pred_mask)

            dice_coef_dict[img['id']] = dice_coef
            cv2.imwrite(os.path.join(self.output_path, image_name), image)
        dice_coef_df = pd.DataFrame().from_dict(dice_coef_dict, orient='index').reset_index()
        dice_coef_df.rename(columns={'index': 'image', 0: 'dice_coef'}, inplace=True)
        store_data_frame(dice_coef_df, os.path.join(self.output_path, 'dice_coef.csv'))


if __name__ == '__main__':
    Main(EvaluationMasksUOC()).run()
