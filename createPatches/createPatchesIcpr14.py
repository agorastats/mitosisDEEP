import logging
import os
import cv2
import numpy as np

from utils.runnable import Main
from createPatches.base import CreatePatches
from utils.image import create_mask_with_annotations_circle

# data images (training data set) from: https://mitos-atypia-14.grand-challenge.org/Donwload/
FOLDERS_NAME = ['A03', 'A04', 'A05', 'A07', 'A10', 'A11', 'A12', 'A14', 'A15', 'A17', 'A18'] + \
               ['H03', 'H04', 'H05', 'H07', 'H10', 'H11', 'H12', 'H14', 'H15', 'H17', 'H18']

DATA_PATH = 'data/icpr14'  # path that contains inside it training folder images


class CreatePatchesIcpr14(CreatePatches):

    def __init__(self):
        super().__init__()
        self.folders_name = FOLDERS_NAME
        self.data_path = DATA_PATH
        self.annot_path = self.data_path
        self.img_format = '.tiff'
        self.patchify = False

    def get_annotations(self, path):
        result = []
        if not os.path.exists(path):
            return result
        ln = 0
        for line in open(path).readlines():
            ln += 1
            # Ignore empty lines
            if len(line.strip()) == 0:
                continue
            # Parse line into list of numbers and extract the two first, last is note of confidence degree
            points = list(map(lambda x: x, line.strip().split(',')))[:2]
            try:
                if len(points) != 2:
                    raise
                result.append(list(map(int, points)))
            except:
                raise Warning("Line %d in %s has invalid value." % (ln, path))
        return result

    def create_patches_with_annotations(self, image, mask, annotations_list, name_img, patch_size):
        # to look patches use: show_image(image_patch), show_image(mask_patch)
        # create patches over annotations list
        for i, m in enumerate(annotations_list):
            centered_at = np.random.RandomState(self.seed_count).uniform(self.centered_limits[0],
                                                                         self.centered_limits[1], 2)
            w, h = m
            image_patch = self.generate_patch(image, h, w, centered_at, patch_size=patch_size)
            mask_patch = self.generate_patch(mask, h, w, centered_at, patch_size=patch_size)
            assert sum(list(image_patch.shape)[:2]) == 2 * patch_size, \
                'Error in expected shape of patch. Check image %s' % str(name_img)
            self.write_patches(image_patch, mask_patch, name_img, i)

    def run(self, options):
        for folder in self.folders_name:
            logging.info('Iterating folder:  %s' % str(folder))
            data_dir = os.path.join(self.data_path, folder, 'frames/x40/')
            annot_dir = os.path.join(self.annot_path, folder, 'mitosis')
            images_list = [f for f in os.listdir(data_dir) if f.endswith(self.img_format)]
            for j, img in enumerate(sorted(images_list)):
                if j % 100 == 0:
                    logging.info('(progress) Iterating image number: %i / %i' % (j, len(images_list)))
                logging.info('___prepare patches for img: %s' % str(img))
                name_img = img.split('.')[0]
                image = cv2.imread(os.path.join(data_dir, img))
                annot_list = self.get_annotations(os.path.join(annot_dir, name_img + '_mitosis.csv'))
                mask = create_mask_with_annotations_circle(image, annot_list)
                assert len(np.unique(mask)) <= 2, 'more than 2 color pixels'

                self.create_patches_with_annotations(image, mask, annot_list, name_img,
                                                     patch_size=options['patch_size'])

                self.create_patches_with_patchify(image, mask, name_img, patch_size=options['patch_size'])


if __name__ == '__main__':
    Main(
        CreatePatchesIcpr14()
    ).run()
