import logging
import os
import cv2
import numpy as np
import pandas as pd
from patchify import patchify

from utils.image import load_annotations, create_dir, create_mask_with_annotations, generate_patch, show_image

# data images (training data set) from: http://ludo17.free.fr/mitos_2012/download.html
FOLDERS_NAME = ['A0%i_v2' % i for i in range(5)] + ['H0%i_v2' % i for i in range(5)]  # training folder images
DATA_PATH = 'icpr12_data'  # path that contains inside it training folder images
PATCH_SIZE = 256  # shape of symmetric patch: (patch_size, patch_size)
IMAGES_FOLDER = 'images_2021_12_16'    # folder to save patches of images
MASKS_FOLDER = 'masks_2021_12_16'     # folder to save patches of masks

if __name__ == '__main__':
    # create needed folders
    output_dir = os.path.join(DATA_PATH, 'patches')
    create_dir(output_dir)
    create_dir(os.path.join(output_dir, IMAGES_FOLDER))
    create_dir(os.path.join(output_dir, MASKS_FOLDER))
    for folder in FOLDERS_NAME:
        data_dir = os.path.join(DATA_PATH, folder)
        images_list = [f for f in os.listdir(data_dir) if f.endswith('.bmp')]
        for j, img in enumerate(images_list):
            logging.info('___prepare patches for img: %s' % str(img))
            name_img = img.split('.')[0]
            image = cv2.imread(os.path.join(data_dir, img))
            annotations_list = load_annotations(os.path.join(data_dir, name_img + '.csv'))
            mask = create_mask_with_annotations(image, annotations_list)
            assert len(np.unique(mask)) <= 2, 'more than 2 color pixels'
            # create patches over annotations list
            for i, m in enumerate(annotations_list):
                w, h = pd.DataFrame(m).mean(axis=0).round(0).astype(int)   # get centroid of mitosis
                image_patch = generate_patch(image, h, w, patch_size=PATCH_SIZE)
                mask_patch = generate_patch(mask, h, w, patch_size=PATCH_SIZE)
                assert sum(list(image_patch.shape)[:2]) == 2*PATCH_SIZE,\
                    'Error in expected shape of patch. Check image %s' % str(name_img)
                cv2.imwrite(os.path.join(output_dir, IMAGES_FOLDER, name_img + '_' + str(i) + '.jpg'), image_patch)
                cv2.imwrite(os.path.join(output_dir, MASKS_FOLDER, name_img + '_' + str(i) + '.jpg'), mask_patch)
                # to look patches
                # show_image(image_patch)
                # show_image(mask_patch)

            # create random patches over all image (using patchify)
            all_patches_img = patchify(image, (256, 256, 3), step=256)  # Step=256 for 256 patches means no overlap
            all_patches_mask = patchify(mask, (256, 256), step=256)
            np.random.seed(j)
            ii = np.random.choice(range(all_patches_img.shape[0]), 2)
            np.random.seed(j+1)
            jj = np.random.choice(range(all_patches_img.shape[1]), 2)
            for s in ii:
                for k in jj:
                    single_patch_img = all_patches_img[s, k, :, :]
                    single_patch_mask = all_patches_mask[s, k, :, :]
                    cv2.imwrite(os.path.join(output_dir, IMAGES_FOLDER, name_img + '_' + str(s) + '_' + str(k) + '.jpg'),  single_patch_img[0])
                    cv2.imwrite(os.path.join(output_dir, MASKS_FOLDER, name_img + '_' + str(s) + '_' + str(k) + '.jpg'),  single_patch_mask)

