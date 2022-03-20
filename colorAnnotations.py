import cv2
import numpy as np

from utils.image import show_image, read_image
from utils.loadAndSaveResults import read_data_frame


if __name__ == '__main__':
    for i in np.arange(1, 36, 1):
        img = cv2.imread("data/UOC Mitosis/0%ix.jpg" % i)
        show_image(img)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([170, 25, 0])
        upper = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

    mean = np.mean(img, axis=(0, 1))
    std = np.std(img, axis=(0, 1))
    mask = (np.abs(img - mean) / std >= 4.5).any(axis=2)
    mask_u8 = mask.astype(np.uint8) * 255
    show_image(mask_u8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = sorted(contours, key=cv2.contourArea, reverse=True)
    # cv2.drawContours(img, [each candidates coords], -1, (255, 0, 0), 3)
    pass


