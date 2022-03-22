import logging
import os
import cv2
import numpy as np

from abc import ABCMeta, abstractmethod
from utils.image import create_dir
from utils.runnable import Runnable

DEF_MASK_OUTPUT_FOLDER = 'masks'


def get_annotations_by_zscore(img, thresh=4.5):
    mean = np.mean(img, axis=(0, 1))
    std = np.std(img, axis=(0, 1))
    mask = (np.abs(img - mean) / std >= thresh).any(axis=2)
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # keep center of contours with minimum shape
    annot_list = [c.mean(axis=0).astype(int).tolist()[0] for c in contours if c.shape[0] > 40]
    return annot_list


class CreateMaskAnnotations(Runnable, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()
        self.data_path = None
        self.images_list = None
        self.mask_output = DEF_MASK_OUTPUT_FOLDER

    def create_needed_folders(self, options):
        create_dir(os.path.join(self.data_path, self.mask_output))

    @abstractmethod
    def get_annotations(self, *args):
        # ref to convert hsv: https://toolstud.io/color/rgb.php?
        pass

    def pre_run(self, options):
        self.create_needed_folders(options)
        logging.info('iterating process: %s' % self.__class__.__name__)

    def run(self, options):
        pass
