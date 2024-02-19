import json
import logging
import os
import cv2
import numpy as np

from createPatches.base import CreatePatches
from utils.image import create_mask_with_annotations_circle, create_mask_with_annotations_polynomial, \
    create_shape_mask_inferring_from_centroid_annotations, read_mask
from utils.runnable import Main


DATA_PATH = 'data/GZMH/train/image'
ANNOT_PATH = 'data/GZMH/train/label'
ANNOT_FILE = 'data/GZMH/train/annot_training.txt'
class CreatePatchesGZMH(CreatePatches):

    def __init__(self):
        super().__init__()
        self.prefix_img = 'GZMH'
        self.data_path = DATA_PATH
        self.annot_path = ANNOT_PATH
        self.img_format = '.jpg'
        self.patchify = True

    def get_annotations(self, path, img_list):
        try:
            img_name_list = [img.replace('.jpg', '') for img in img_list]
            with open(path, 'r') as file:
                annot_lines = file.readlines()
                annot_dict_list = []
                for i, line in enumerate(annot_lines[:-1]):
                    img, mitosis_num = line.strip().replace('\t', ' ').split(' ')[:2]
                    if img in img_name_list:
                        aux_dict = {'img': img, 'mitosis_num': int(mitosis_num)}
                        for k in range(0, 2*int(mitosis_num)+2, 2)[1:]: # iter every 2 to get only mean of centroid
                            if 'centroids' not in aux_dict:
                                aux_dict['centroids'] = []
                            centroid_vals = [int(float(c)) for c in \
                                             annot_lines[i+k].strip().replace('\t', ' ').split(' ')][::-1] # reverse coords
                            aux_dict['centroids'].append(centroid_vals)
                        annot_dict_list.append(aux_dict)
        except:
            raise Warning("problems reading annotations ")
        return annot_dict_list

    def run(self, options):
        data_dir = self.data_path
        annot_dir = self.annot_path
        img_format = self.img_format
        images_list = [f for f in os.listdir(data_dir) if f.endswith(self.img_format)]
        annot_dict_list = self.get_annotations(ANNOT_FILE, images_list)
        for j, annot_dict in enumerate(annot_dict_list):
            if j % 100 == 0:
                logging.info('(progress) Iterating image number: %i / %i' % (j, len(annot_dict_list)))
            name_img = annot_dict['img']
            logging.info('___prepare patches for img: %s' % str(name_img))
            img_file = '{}{}'.format(name_img, img_format)
            image = cv2.imread(os.path.join(data_dir, img_file))
            mask = read_mask(os.path.join(annot_dir, img_file))
            assert len(np.unique(mask)) <= 2, 'more than 2 color pixels'
            self.create_patches_with_annotations(image, mask, annot_dict['centroids'], name_img,
                                                 patch_size=options['patch_size'])
            self.create_patches_with_patchify(image, mask, name_img, patch_size=options['patch_size'])


if __name__ == '__main__':
    Main(
        CreatePatchesGZMH()
    ).run()
