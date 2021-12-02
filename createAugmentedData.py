import logging
import os

import cv2

from utils.augmentation import aug_generator, stain_normalizer
from utils.image import create_dir, read_and_normalize_image

# path that contains inside it two folders with patches: images, masks
DATA_PATH = 'icpr12_data/patches'
AUG_VALUE = 5
REF_IMAGE = 'utils/ref_he.jpg'
STAIN_NORM = True

if __name__ == '__main__':
    # create folders for augmented data
    images_aug = os.path.join(DATA_PATH, 'images_aug2')
    masks_aug = os.path.join(DATA_PATH, 'masks_aug2')
    create_dir(images_aug)
    create_dir(masks_aug)
    images_list = [f for f in os.listdir(os.path.join(DATA_PATH, 'images')) if f.endswith('.jpg')]
    # reinhard stain normalization
    stain_norm = stain_normalizer(REF_IMAGE) if STAIN_NORM else None
    for i, img in enumerate(images_list):
        logging.info('___create augmented patches for img: %s' % str(img))
        name_img = img.split('.')[0]
        image = cv2.imread(os.path.join(DATA_PATH, 'images', img))
        assert img is not None, 'problems reading image %s' % str(img)
        mask = cv2.imread(os.path.join(DATA_PATH, 'masks', img))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        assert mask is not None, 'problems reading mask of image %s' % str(img)
        for j in range(AUG_VALUE+1):
            aug_image, aug_mask = aug_generator(image, mask, seed=i+j, stain_norm=stain_norm)
            cv2.imwrite(os.path.join(images_aug, name_img + '_' + str(j) + '.jpg'), aug_image)
            cv2.imwrite(os.path.join(masks_aug, name_img + '_' + str(j) + '.jpg'), aug_mask)







