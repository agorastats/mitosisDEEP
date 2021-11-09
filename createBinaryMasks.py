import logging
import os
import re
import cv2
import imutils as imutils

from utils.reader import load_annotations, create_output_dir, create_mask_with_annotations

if __name__ == '__main__':

    output_dir = 'sample_data/proves/prova1_A00_v2/'
    create_output_dir(output_dir)

    data_dir = 'sample_data/A00_v2'
    images_list = [f for f in os.listdir(data_dir) if f.endswith('.bmp')]
    for img in images_list:
        logging.info('___prepare mask for img: %s' % str(img))
        name_img = img.split('.')[0]
        image = cv2.imread(os.path.join(data_dir, img))
        annotations_list = load_annotations(os.path.join(data_dir, name_img + '.csv'))
        mask_image = create_mask_with_annotations(image, annotations_list)
        cv2.imwrite(os.path.join(output_dir, 'masks', name_img + '.jpg'), mask_image)
        cv2.imwrite(os.path.join(output_dir, 'images', name_img + '.jpg'), image)


