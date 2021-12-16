import logging
import os

import cv2
import pandas as pd

from utils.augmentation import aug_generator, stain_normalizer, TRANSFORM_PROB_DICT, TRANSFORM_COLS
from utils.image import create_dir, read_and_normalize_image

# path that contains inside it two folders with patches: images, masks
DATA_PATH = 'icpr12_data/patches'
AUG_VALUE = 5
REF_IMAGE = 'utils/ref_he.jpg'
STAIN_NORM = True
OUT_IMAGES_FOLDER = 'images_aug_2021_12_15'    # folder to save patches of images
OUT_MASKS_FOLDER = 'masks_aug_2021_12_15'     # folder to save patches of masks
INPUT_IMAGES_FOLDER = 'images_2021_12_15'             # folder inside DATA_PATH containing images patches
INPUT_MASKS_FOLDER = 'masks_2021_12_15'               # folder inside DATA_PATH containing masks patches


def update_augmentations_on_image_info(info_aug_df_list, name_aug, prob):
    aux_dict = {k: prob < TRANSFORM_PROB_DICT[k] for k in TRANSFORM_COLS}
    aux_df = pd.DataFrame(aux_dict, index=[0])
    aux_df.loc[:, 'name'] = pd.Series(name_aug, index=aux_df.index)
    info_aug_df_list.append(pd.DataFrame(aux_df))


if __name__ == '__main__':
    # create folders for augmented data
    images_aug = os.path.join(DATA_PATH, OUT_IMAGES_FOLDER)
    masks_aug = os.path.join(DATA_PATH, OUT_MASKS_FOLDER)
    create_dir(images_aug)
    create_dir(masks_aug)
    info_aug_df_list = [pd.DataFrame(columns=TRANSFORM_COLS + ['name'])]
    images_list = [f for f in os.listdir(os.path.join(DATA_PATH, INPUT_IMAGES_FOLDER)) if f.endswith('.jpg')]
    # reinhard stain normalization
    stain_norm = stain_normalizer(REF_IMAGE) if STAIN_NORM else None
    for i, img in enumerate(images_list):
        if i % 100 == 0:
            print("Iterating patch number: %i / %i" % (i, len(images_list)))
        logging.info('___create augmented patches for img: %s' % str(img))
        name_img = img.split('.')[0]

        image = cv2.imread(os.path.join(DATA_PATH, INPUT_IMAGES_FOLDER, img))
        assert image is not None, 'problems reading image %s' % str(img)

        mask = cv2.imread(os.path.join(DATA_PATH, INPUT_MASKS_FOLDER, img))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        assert mask is not None, 'problems reading mask of image %s' % str(img)

        # real patches
        cv2.imwrite(os.path.join(images_aug, name_img + '.jpg'), image)
        cv2.imwrite(os.path.join(masks_aug, name_img + '.jpg'), mask)

        # augmented patches
        for j in range(AUG_VALUE):
            name_aug = name_img + '_' + str(j)
            aug_image, aug_mask, p = aug_generator(image, mask, seed=i+j, stain_norm=stain_norm)
            cv2.imwrite(os.path.join(images_aug, name_aug + '.jpg'), aug_image)
            cv2.imwrite(os.path.join(masks_aug, name_aug + '.jpg'), aug_mask)
            update_augmentations_on_image_info(info_aug_df_list, name_aug, p)

    # save augmentations notes
    aug_notes_df = pd.concat(info_aug_df_list, ignore_index=True)
    aug_notes_df.to_csv('../' + OUT_IMAGES_FOLDER + '_notes.csv', sep=';', index=False)






