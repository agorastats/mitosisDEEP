import os
import cv2
import numpy as np
from skimage.measure import label, regionprops
from utils.loadAndSaveResults import read_data_frame
from utils.image import trace_boundingBox, maskInColor



if __name__ == '__main__':
    from utils.image import show_image, rle_decode

    # https://stackoverflow.com/questions/65138694/opencv-blob-defect-anomaly-detection
    img = cv2.imread("sample_data/A01_09.jpg")
    show_image(img)
    mean = np.mean(img, axis=(0, 1))
    std = np.std(img, axis=(0, 1))
    mask = (np.abs(img - mean) / std >= 4.5).any(axis=2)
    mask_u8 = mask.astype(np.uint8) * 255
    # show_image(mask_u8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = sorted(contours, key=cv2.contourArea, reverse=True)
    # cv2.drawContours(img, [candidates coords], -1, (255, 0, 0), 3)

    testDF = read_data_frame('sample_data/pred_info.csv')
    test = testDF.loc[testDF.loc[:, 'id'] == 'A01_09.bmp', :].squeeze()
    img = cv2.resize(img, (test['size_x'], test['size_y']))
    pred_mask = rle_decode(test['rle'], np.array((test['size_y'], test['size_x'])))
    show_image(trace_boundingBox(img, pred_mask))
    pass

