import json
import logging
import os
import cv2
import numpy as np

from createPatches.base import CreatePatches
from utils.image import create_mask_with_annotations_circle, create_mask_with_annotations_polynomial
from utils.runnable import Main

# data images (training data set) from: BreCaHAD: A Dataset for Breast Cancer Histopathological Annotation and Diagnosis
FOLDERS_NAME = ['images']  # training folder images
DATA_PATH = 'data/BreCaHAD'  # path that contains inside it training folder images
ANNOT_PATH = 'data/BreCaHAD/groundTruth'


class CreatePatchesBreCaHad(CreatePatches):

    def __init__(self):
        super().__init__()
        self.folders_name = FOLDERS_NAME
        self.data_path = DATA_PATH
        self.annot_path = ANNOT_PATH
        self.img_format = '.tif'

    def get_annotations(self, path, image_shape):
        try:
            # opening JSON file
            f = open(path)
            data = json.load(f)
            annot_dict_list = data.get('mitosis')
            f.close()
            annot_list = [list(annot_dict_list[k].values()) for k in range(len(annot_dict_list))]
            # renormalize annotations based on image shape (are between [0,1])
            for i, annot in enumerate(annot_list):
                annot_list[i][0] = int(annot[0] * image_shape[1])
                annot_list[i][1] = int(annot[1] * image_shape[0])
        except:
            raise Warning("problems reading annotations json")
        return annot_list

    def create_patches_with_annotations(self, image, mask, annotations_list, name_img, patch_size):
        # to look patches use: show_image(image_patch), show_image(mask_patch)
        # create patches over annotations list
        for i, m in enumerate(annotations_list):
            w, h = m
            centered_at = np.random.RandomState(self.seed_count).uniform(1, 3, 2)  # to avoid centered patches
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
            images_list = [f for f in os.listdir(data_dir) if f.endswith(self.img_format)]
            for j, img in enumerate(sorted(images_list)):
                if j % 100 == 0:
                    logging.info('(progress) Iterating image number: %i / %i' % (j, len(images_list)))
                logging.info('___prepare patches for img: %s' % str(img))
                name_img = img.split('.')[0]
                image = cv2.imread(os.path.join(data_dir, img))
                annot_list = self.get_annotations(os.path.join(annot_dir, name_img + '.json'), image.shape)
                mask = create_mask_with_annotations_circle(image, annot_list)
                assert len(np.unique(mask)) <= 2, 'more than 2 color pixels'

                self.create_patches_with_annotations(image, mask, annot_list, name_img,
                                                     patch_size=options['patch_size'])

                self.create_patches_with_patchify(image, mask, name_img, patch_size=options['patch_size'])


if __name__ == '__main__':
    Main(
        CreatePatchesBreCaHad()
    ).run()
