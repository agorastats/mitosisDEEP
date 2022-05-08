import logging
import os
import cv2
import numpy as np
import pandas as pd

from maskAnnotations.base import CreateMaskAnnotations
from utils.runnable import Main
from createPatches.base import CreatePatches
from utils.image import create_mask_with_annotations_polynomial

# ref: http://ludo17.free.fr/mitos_2012/download.html
FOLDERS_NAME = ['A0%i_v2' % i for i in range(5)] + ['H0%i_v2' % i for i in range(5)]  # folder images
DATA_PATH = 'data/icpr12/validation_data'  # path that contains inside eval folder images


class CreateMaskAnnotationsICPR12Eval(CreateMaskAnnotations):

    def __init__(self):
        super().__init__()
        self.folders_name = FOLDERS_NAME
        self.data_path = DATA_PATH
        self.annot_path = self.data_path
        self.img_format = '.bmp'

    def get_annotations(self, path):
        result = []
        ln = 0
        for line in open(path).readlines():
            ln += 1
            # Ignore empty lines
            if len(line.strip()) == 0:
                continue
            # Parse line into list of numbers
            points = list(map(lambda x: x, line.strip().split(',')))
            try:
                result.append([[int(points[i]), int(points[i + 1])] for i in range(0, len(points), 2)])
            except:
                raise Warning("Line %d in %s has invalid value." % (ln, path))
        return result

    def create_mask_with_annotations(self, image, annot_list):
        mask = create_mask_with_annotations_polynomial(image, annot_list)
        return mask

    def run(self, options):
        for folder in self.folders_name:
            logging.info('Iterating folder:  %s' % str(folder))
            data_dir = os.path.join(self.data_path, folder)
            annot_dir = os.path.join(self.annot_path, folder)
            images_list = [f for f in os.listdir(data_dir) if f.endswith(self.img_format)]
            for j, img in enumerate(sorted(images_list)):
                if j % 100 == 0:
                    logging.info('(progress) Iterating image number: %i / %i' % (j, len(images_list)))
                logging.info('___prepare annot mask for img: %s' % str(img))
                name_img = img.split('.')[0]
                image = cv2.imread(os.path.join(data_dir, img))
                annot_list = self.get_annotations(os.path.join(annot_dir, name_img + '.csv'))
                mask = self.create_mask_with_annotations(image, annot_list)
                assert len(np.unique(mask)) <= 2, 'more than 2 color pixels'
                # save mask as jpg
                cv2.imwrite(os.path.join(self.data_path, self.mask_output, img.replace('.bmp', '.jpg')), mask)


if __name__ == '__main__':
    Main(
        CreateMaskAnnotationsICPR12Eval()
    ).run()
