import logging
import os
import cv2
import numpy as np

from createPatches.base import CreatePatches
from utils.image import create_mask_with_annotations_circle, create_shape_mask_inferring_from_centroid_annotations
from utils.runnable import Main

# AMIDA13 challenge. # These cases were collected from the Department of Pathology
# at the University Medical Center in Utrecht, The Netherlands.

FOLDERS_NAME = ['mitoses_image_data_part_1', 'mitoses_image_data_part_2', 'mitoses_image_data_part_3']
DATA_PATH = 'data/AMIDA13'  # path that contains inside it training folder images
ANNOT_PATH = 'data/AMIDA13/mitoses_ground_truth'


class CreatePatchesAmida13(CreatePatches):

    def __init__(self):
        super().__init__()
        self.folders_name = FOLDERS_NAME
        self.data_path = DATA_PATH
        self.annot_path = ANNOT_PATH
        self.img_format = '.tif'
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
            points = [int(p) for p in points][::-1]  # reverse it, interpret as rows (width), columns (height)
            try:
                result.append(points)
            except:
                raise Warning("Line %d in %s has invalid value." % (ln, path))
        return result

    def create_patches_with_annotations(self, image, mask, annotations_list, name_img, patch_size):
        # to look patches use: show_image(image_patch), show_image(mask_patch)
        # create patches over annotations list
        for i, m in enumerate(annotations_list):
            w, h = m
            centered_at = self.get_center_positions()
            image_patch = self.generate_patch(image, h, w, centered_at, patch_size=patch_size)
            mask_patch = self.generate_patch(mask, h, w, centered_at, patch_size=patch_size)
            assert sum(list(image_patch.shape)[:2]) == 2 * patch_size, \
                'Error in expected shape of patch. Check image %s' % str(name_img)
            self.write_patches(image_patch, mask_patch, name_img, i)

    def run(self, options):
        for folder in self.folders_name:
            logging.info('Iterating folder:  %s' % str(folder))
            data_dir = os.path.join(self.data_path, folder)
            annot_dir = os.path.join(self.annot_path)
            # image regions in folder
            regions_list = [f for f in os.listdir(data_dir)]
            for r, region in enumerate(sorted(regions_list)):
                logging.info('Iterating region folder:  %i / %i' % (r, len(regions_list)))
                region_dir = os.path.join(data_dir, region)
                images_list = [f for f in os.listdir(region_dir) if f.endswith(self.img_format)]
                for j, img in enumerate(sorted(images_list)):
                    if j % 100 == 0:
                        logging.info('(progress) Iterating image number: %i / %i' % (j, len(images_list)))
                    logging.info('__prepare patches for img: %s' % str(img))
                    name_img = img.split('.')[0]
                    image = cv2.imread(os.path.join(region_dir, img))
                    if name_img + '.csv' not in os.listdir(os.path.join(annot_dir, region)):
                        logging.info('___not annotations for img: %s' % str(img))
                        continue
                    annot_list = self.get_annotations(os.path.join(annot_dir, region, name_img + '.csv'))
                    # mask = create_mask_with_annotations_circle(image, annot_list, radius=50)
                    mask = create_shape_mask_inferring_from_centroid_annotations(image, annot_list)
                    assert len(np.unique(mask)) <= 2, 'more than 2 color pixels'

                    aux_name_img = 'region_' + region + '_' + name_img
                    self.create_patches_with_annotations(image, mask, annot_list, aux_name_img,
                                                         patch_size=options['patch_size'])

                    self.create_patches_with_patchify(image, mask, aux_name_img, patch_size=options['patch_size'])


if __name__ == '__main__':
    Main(
        CreatePatchesAmida13()
    ).run()
