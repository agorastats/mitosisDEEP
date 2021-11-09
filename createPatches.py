import logging
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff

from preprocessing.stainNorm_Reinhard import Normalizer
from utils.reader import create_output_dir

if __name__ == '__main__':
    # test with data: A00_v2
    output_dir = 'sample_data/proves/patch_A00_v2/'  # contains images and masks
    create_output_dir(output_dir)
    # create folder for images and masks
    create_output_dir(output_dir + 'images/')
    create_output_dir(output_dir + 'masks/')

    data_dir = 'sample_data/proves/prova1_A00_v2'
    images_list = [f for f in os.listdir(os.path.join(data_dir, 'images'))]
    masks_list = images_list
    stainNormalizer = Normalizer()

    logging.info('__preparing patches for images')
    for img in images_list:
        name = img.split('.')[0]
        large_image = cv2.imread(os.path.join(data_dir, 'images', img))
        large_image = cv2.cvtColor(large_image, cv2.COLOR_BGR2RGB)
        # apply stain normalization
        stainNormalizer.fit(large_image)
        large_image = stainNormalizer.transform(large_image)
        # large_image is 2048x2048, so we obtain 8 patches of 256x256
        patches_img = patchify(large_image, (256, 256, 3), step=256)  # Step=256 for 256 patches means no overlap

        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i, j, :, :]
                tiff.imwrite(output_dir + 'images/' + str(name) + '_' + str(i) + str(j) + ".tif", single_patch_img)

    logging.info('__preparing patches for masks')
    for msk in images_list:
        name = msk.split('.')[0]
        large_mask = cv2.imread(os.path.join(data_dir, 'masks', msk))
        patches_mask = patchify(large_mask, (256, 256, 3), step=256)  # Step=256 for 256 patches means no overlap
        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):
                single_patch_mask = patches_mask[i, j, :, :]
                tiff.imwrite(output_dir + '/masks/' + str(msk) + '_' + str(i) + str(j) + ".tif", single_patch_mask)
                single_patch_mask = single_patch_mask / 255.
