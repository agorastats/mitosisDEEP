import logging
import os
import sys
import uuid

import cv2

from utils.image import read_image, create_dir, show_image
from utils.runnable import Runnable, Main
from utils.stain.preprocessStain import Normalizer, normalize_staining

IMG_PATH_KEY = 'imgPathKey'


class StainApplyToPatches(Runnable):

    def __init__(self):
        super().__init__()
        self.path = None

    def add_options(self, parser):
        super().add_options(parser)
        parser.add_option('--imagesPath', dest=IMG_PATH_KEY,
                          help='Path containing images to apply HE normalizer',
                          action='store', default=None)

    def pre_run(self, options):
        assert options[IMG_PATH_KEY] is not None, 'need to fill path containing patches!'
        self.path = os.path.join(os.path.dirname(options[IMG_PATH_KEY]), 'he_images_' + str(uuid.uuid4()))
        create_dir(self.path)

    def run(self, options):
        img_list = os.listdir(options[IMG_PATH_KEY])
        for j, patch in enumerate(img_list):
            if j % 100 == 0:
                logging.info('(progress) Iterating image number: %i / %i' % (j, len(img_list)))
            img = read_image(os.path.join(options[IMG_PATH_KEY], patch))  # read and convert to rgb format
            try:
                img = normalize_staining(img)  # apply HE
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # need to change to rewrite correctly
                cv2.imwrite(os.path.join(self.path, patch), img)  # rewrite it
            except:
                logging.info('problems with patch: %s' % patch)


if __name__ == '__main__':
    args = sys.argv[1:]
    args += ['--imagesPath', 'patches/all_data/images']
    Main(
        StainApplyToPatches()
    ).run(args)
