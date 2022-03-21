import logging
import os

from abc import ABCMeta, abstractmethod

import numpy as np

from utils.image import create_dir
from utils.loadAndSaveResults import read_data_frame
from utils.runnable import Runnable


def get_dice_coef(mask1, mask2):
    intersect = np.sum(mask1 * mask2)
    fsum = np.sum(mask1)
    ssum = np.sum(mask2)
    dice = (2 * intersect) / (fsum + ssum)
    dice = np.mean(dice).round(3)
    return dice


INFO_CSV_KEY = 'info_csv'


class EvaluationMasks(Runnable, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()
        self.data_path = None
        self.mask_path = None
        self.pred_info = None
        self.images_list = None
        self.suffix_annot = ''
        self.output_path = None

    def add_options(self, parser):
        super().add_options(parser)
        parser.add_option('--infoCsv', dest=INFO_CSV_KEY,
                          help='Folder containing rle encode prediction masks info (csv format)',
                          action='store', default=None)

    def create_needed_folders(self, options):
        info_csv = options[INFO_CSV_KEY]
        info_csv_name = info_csv.split('.')[0]
        self.output_path = os.path.join(self.data_path, info_csv_name) if self.output_path is None else self.output_path
        create_dir(self.output_path)

    def get_pred_info(self, options):
        infoDF = read_data_frame(os.path.join(self.data_path, options[INFO_CSV_KEY]))
        infoDF.loc[:, 'rle'].fillna('', inplace=True)
        return infoDF

    def pre_run(self, options):
        assert options[INFO_CSV_KEY] is not None, 'need to inform pred_info file'
        self.create_needed_folders(options)
        logging.info('iterating process: %s' % self.__class__.__name__)

    def run(self, options):
        pass
