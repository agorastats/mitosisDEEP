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
        name_img += '_' + str(number_patch) + '.jpg'
        cv2.imwrite(os.path.join(self.img_output, name_img), img_patch)
        cv2.imwrite(os.path.join(self.mask_output, name_img), mask_patch)
        self.patches_list.append(name_img)
        self.seed_count += 1

    def create_patches_with_patchify(self, image, mask, name_img, patch_size=256, n_patches=2):
        if self.patchify:
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

    @abstractmethod
    def create_patches_with_annotations(self, image, mask, annotations, name_img, patch_size=256):
        pass

    def apply_centroid_to_shape_mask(self, image, mask):
        # kernel to erode image
        kernel = np.ones((3, 3), np.uint8)
        # apply kernel to image
        # erosion_img = cv2.erode(image, kernel, iterations=1)
        # focus erosion_img only over mask pixels
        erosion_img = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=3)
        erosion_img = cv2.morphologyEx(erosion_img, cv2.MORPH_CLOSE, kernel, iterations=3)
        # prepare data to fit local cluster (in pixel-wise mask)
        masked_image = cv2.bitwise_and(erosion_img, erosion_img, mask=mask.astype(bool).astype(np.uint8))
        eroded_image_flattened = masked_image.reshape((-1, 3))
        kmeans = KMeans(n_clusters=2) # two groups, mitosis or not mitosis
        kmeans.fit(eroded_image_flattened)
        labels = kmeans.labels_
        labelMito = np.bincount(labels).argmin()  # assume that mitosis group pixel count is < background pixel count
        segmented_image = labels.reshape(masked_image.shape[:2])
        # inference pixel-wise mask
        shape_mask = np.zeros_like(mask, dtype=np.uint8)
        shape_mask[segmented_image == labelMito] = 255
        final_mask = shape_mask != mask  # inference mask pixels not in prior mask
        return final_mask.astype(np.uint8) * 255

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
