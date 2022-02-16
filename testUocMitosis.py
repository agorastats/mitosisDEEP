import os

import cv2
import numpy as np

from utils.loadAndSaveResults import read_data_frame
from utils.image import trace_boundingBox, rle_decode, create_dir

OUTPUT_PATH = 'data/UOC Mitosis/pred'
PRED_INFO_CSV = 'pred_info_unet_freeze'

if __name__ == '__main__':
    create_dir(OUTPUT_PATH)
    testDF = read_data_frame('data/UOC Mitosis/%s' % str(PRED_INFO_CSV))
    testDF.loc[:, 'rle'].fillna('', inplace=True)
    for i, img in testDF.iterrows():
        image_name = img['id'].split('.')[0] + 'x.jpg'
        image = cv2.imread(os.path.join('data/UOC Mitosis', image_name))
        if len(img['rle']) > 0:
            image = cv2.resize(image, (img['size_x'], img['size_y']))
            pred_mask = rle_decode(img['rle'], np.array((img['size_y'], img['size_x'])))
            image = trace_boundingBox(image, pred_mask, color=(0, 0, 0), width=15)
        cv2.imwrite(os.path.join(OUTPUT_PATH, image_name), image)
    pass

