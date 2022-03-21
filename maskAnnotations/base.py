import logging
import os

from abc import ABCMeta, abstractmethod
from utils.image import create_dir
from utils.runnable import Runnable

DEF_MASK_OUTPUT_FOLDER = 'masks'


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
