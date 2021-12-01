import os

import cv2
import numpy as np

from utils.image import read_and_normalize_image, create_dir

IMAGES_PATH = 'icpr12_data/patches/images_aug'
MASK_PATH = 'icpr12_data/patches/masks_aug'
OUTPUT_NPY_FILES = 'icpr12_data/patches/npy_files'

if __name__ == '__main__':
    # save images, masks as npy files
    images_files_list = [f for f in os.listdir(IMAGES_PATH) if f.endswith('.jpg')]
    images_list = list()
    for image in images_files_list:
        img = read_and_normalize_image(os.path.join(IMAGES_PATH, image))
        images_list.append(img)

    masks_files_list = [f for f in os.listdir(MASK_PATH) if f.endswith('.jpg')]
    masks_list = list()
    for mask in masks_files_list:
        msk = read_and_normalize_image(os.path.join(MASK_PATH, mask), convert=cv2.COLOR_BGR2GRAY)
        masks_list.append(msk)

    # save lists as npy files
    create_dir(OUTPUT_NPY_FILES)
    np.save(os.path.join(OUTPUT_NPY_FILES, 'images.npy'), np.array(images_list))
    np.save(os.path.join(OUTPUT_NPY_FILES, 'masks.npy'), np.array(masks_list))