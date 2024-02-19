import logging
import os
import cv2
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from patchify import patchify
from abc import ABCMeta, abstractmethod
from datetime import date, datetime
from utils.image import create_dir
from utils.loadAndSaveResults import store_data_frame
from utils.runnable import Runnable

DEF_PATCHES_DIR = 'patches'


class CreatePatches(Runnable, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()
        self.prefix_img = None # prefix to all images indicating origin dataset
        self.data_path = None
        self.annot_path = None
        self.patches_dir = DEF_PATCHES_DIR
        self.img_output = None
        self.mask_output = None
        self.df_info_output = None
        self.folders_name = None
        self.img_format = None
        self.patches_list = list()
        self.patchify = True
        self.centered_limits = [1.75, 2.25]   # to avoid centered patches
        self.seed_count = 1

    def add_options(self, parser):
        parser.add_option('--output', dest='output',
                          help='Folder to save images, masks', action='store', default=date.today().strftime("%Y%m%d"))
        parser.add_option('--patchSize', dest='patch_size',
                          help='Size of patch. Construct height=width patches', type='int', default=256)

    def create_needed_folders(self, options):
        output_dir = os.path.join(self.patches_dir)
        create_dir(output_dir)
        self.img_output = os.path.join(output_dir, options['output'], 'images')
        self.mask_output = os.path.join(output_dir, options['output'], 'masks')
        self.df_info_output = os.path.join(output_dir, options['output'])
        create_dir(self.img_output)
        create_dir(self.mask_output)

    def generate_patch(self, image, height, width, centered, patch_size=256):
        patch_center = np.array([height, width])
        limit_image = image.shape
        # center patch if possible, else contains patch that starts at border of image
        patch_x = int(patch_center[0] - patch_size / centered[0]) if patch_center[0] > patch_size / centered[0] else 0
        patch_y = int(patch_center[1] - patch_size / centered[1]) if patch_center[1] > patch_size / centered[1] else 0
        # to ensure patch size
        patch_x -= max(0, patch_x + patch_size - limit_image[0])
        patch_y -= max(0, patch_y + patch_size - limit_image[1])
        patch_image = image[patch_x:patch_x + patch_size, patch_y:patch_y + patch_size]
        return patch_image

    def write_patches(self, img_patch, mask_patch, name_img, number_patch):
        name_img = self.prefix_img + '_' + name_img + '_' + str(number_patch) + '.jpg'
        cv2.imwrite(os.path.join(self.img_output, name_img), img_patch)
        cv2.imwrite(os.path.join(self.mask_output, name_img), mask_patch)
        self.patches_list.append(name_img)
        self.seed_count += 1

    def create_patches_with_patchify(self, image, mask, name_img, patch_size=256, n_patches=1):
        if self.patchify:
            name_img = self.prefix_img + '_' + name_img  # add prefix of dataset to image
            # create random patches over all image (using patchify)  | step=path_size means no overlap of images
            all_patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)
            all_patches_mask = patchify(mask, (patch_size, patch_size), step=patch_size)
            i = np.random.RandomState(self.seed_count).choice(range(all_patches_img.shape[0]), n_patches)
            j = np.random.RandomState(self.seed_count + 1).choice(range(all_patches_img.shape[1]),
                                                                  max(1, n_patches - 1))
            for s in i:
                for k in j:
                    single_patch_img = all_patches_img[s, k, :, :]
                    single_patch_mask = all_patches_mask[s, k, :, :]
                    cv2.imwrite(os.path.join(self.img_output, name_img + '_' + str(s) + '_' + str(k) + '.jpg'),
                                single_patch_img[0])
                    cv2.imwrite(os.path.join(self.mask_output, name_img + '_' + str(s) + '_' + str(k) + '.jpg'),
                                single_patch_mask)
                    self.patches_list.append(name_img + '_' + str(s) + '_' + str(k) + '.jpg')
            self.seed_count += 1

    @abstractmethod
    def get_annotations(self, *args):
        pass

    def create_patches_with_annotations(self, image, mask, annot_list, name_img, patch_size=None):
        # to look patches use: show_image(image_patch), show_image(mask_patch)
        # create patches over annot_list
        assert patch_size is not None, 'need to inplace patch_size parameter'
        for i, annot in enumerate(annot_list):
            image_patch, mask_patch = self.create_patch(image, annot, mask, name_img, patch_size)
            self.write_patches(image_patch, mask_patch, name_img, i)

    def create_patch(self, image, annot, mask, name_img, patch_size):
        centered_at = self.get_center_positions()
        w, h = annot
        image_patch = self.generate_patch(image, h, w, centered_at, patch_size=patch_size)
        mask_patch = self.generate_patch(mask, h, w, centered_at, patch_size=patch_size)
        assert sum(list(image_patch.shape)[:2]) == 2 * patch_size, \
            'Error in expected shape of patch. Check image %s' % str(name_img)
        return image_patch, mask_patch

    def get_center_positions(self):
        centered_at = np.random.RandomState(self.seed_count).uniform(self.centered_limits[0],
                                                                     self.centered_limits[1], 2)
        return centered_at

    def pre_run(self, options):
        self.create_needed_folders(options)
        logging.info('iterating process: %s' % self.__class__.__name__)

    def run(self, options):
        pass

    def post_run(self, options):
        logging.info('__update patches information: infoDF.csv')
        infoDF = pd.DataFrame(columns=['id'])
        infoDF.loc[:, 'id'] = self.patches_list
        infoDF.loc[:, 'images_from'] = self.data_path
        infoDF.loc[:, 'insertionAt'] = pd.Series(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), index=infoDF.index)
        store_data_frame(infoDF, os.path.join(self.df_info_output, 'infoDF.csv'), mode='a')
        logging.info('Updating patches ids to info df')
