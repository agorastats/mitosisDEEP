import logging
import os
import cv2
import numpy as np
import pandas as pd

from utils.runnable import Main
from createPatches.base import CreatePatches
from utils.image import create_mask_with_annotations_polynomial

# data images (training data set) from: http://ludo17.free.fr/mitos_2012/download.html
FOLDERS_NAME = ['A0%i_v2' % i for i in range(5)] + ['H0%i_v2' % i for i in range(5)]  # training folder images
DATA_PATH = 'data/icpr12'  # path that contains inside it training folder images


class CreatePatchesIcpr12(CreatePatches):

    def __init__(self):
        super().__init__()
        self.prefix_img = 'icpr12'
        self.folders_name = FOLDERS_NAME
        self.data_path = DATA_PATH
        self.annot_path = self.data_path
        self.img_format = '.bmp'
        self.patchify = False

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

    def create_patches_with_annotations(self, image, mask, annotations_list, name_img, patch_size):
        # to look patches use: show_image(image_patch), show_image(mask_patch)
        # create patches over annotations list
        for i, m in enumerate(annotations_list):
            # to avoid centered patches
            centered_at = self.get_center_positions()
            w, h = pd.DataFrame(m).mean(axis=0).round(0).astype(int)  # get centroid of mitosis
            image_patch = self.generate_patch(image, h, w, centered_at, patch_size=patch_size)
            mask_patch = self.generate_patch(mask, h, w, centered_at, patch_size=patch_size)
            assert sum(list(image_patch.shape)[:2]) == 2 * patch_size, \
                'Error in expected shape of patch. Check image %s' % str(name_img)
            self.write_patches(image_patch, mask_patch, name_img, i)

    def run(self, options):
        for folder in self.folders_name:
            logging.info('Iterating folder:  %s' % str(folder))
            data_dir = os.path.join(self.data_path, folder)
            annot_dir = os.path.join(self.annot_path, folder)
            images_list = [f for f in os.listdir(data_dir) if f.endswith(self.img_format)]
            for j, img in enumerate(sorted(images_list)):
                if j % 100 == 0:
                    logging.info('(progress) Iterating image number: %i / %i' % (j, len(images_list)))
                logging.info('___prepare patches for img: %s' % str(img))
                name_img = img.split('.')[0]
                image = cv2.imread(os.path.join(data_dir, img))
                annot_list = self.get_annotations(os.path.join(annot_dir, name_img + '.csv'))
                mask = create_mask_with_annotations_polynomial(image, annot_list)
                assert len(np.unique(mask)) <= 2, 'more than 2 color pixels'

                self.create_patches_with_annotations(image, mask, annot_list, name_img, patch_size=options['patch_size'])

                self.create_patches_with_patchify(image, mask, name_img, patch_size=options['patch_size'])


if __name__ == '__main__':
    Main(
        CreatePatchesIcpr12()
    ).run()



