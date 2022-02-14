import os
import cv2
import numpy as np
from skimage.measure import label, regionprops

from utils.loadAndSaveResults import read_data_frame

if __name__ == '__main__':
    from utils.image import show_image, rle_decode

    # https://stackoverflow.com/questions/65138694/opencv-blob-defect-anomaly-detection
    img = cv2.imread("sample_data/30x.jpg")
    show_image(img)
    mean = np.mean(img, axis=(0, 1))
    std = np.std(img, axis=(0, 1))
    mask = (np.abs(img - mean) / std >= 4.5).any(axis=2)
    mask_u8 = mask.astype(np.uint8) * 255
    show_image(mask_u8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = sorted(contours, key=cv2.contourArea, reverse=True)
    # cv2.drawContours(img, [candidates coords], -1, (255, 0, 0), 3)

    testDF = read_data_frame('sample_data/pred_info.csv')
    test = testDF.loc[testDF.loc[:, 'id'] == '30.jpg', :].squeeze()
    img = cv2.resize(img, (test['size_x'], test['size_y']))
    pred_mask = rle_decode(test['rle'], np.array((test['size_y'], test['size_x'])))


    def maskInColor(image: np.ndarray,
                    mask: np.ndarray,
                    color: tuple = (0, 0, 255),
                    alpha: float = 0.2) -> np.ndarray:
        image = np.array(image)
        H, W, C = image.shape
        mask = mask.reshape(H, W, 1)
        overlay = image.astype(np.float32)
        overlay = 255 - (255 - overlay) * (1 - mask * alpha * color / 255)
        overlay = np.clip(overlay, 0, 255)
        overlay = overlay.astype(np.uint8)
        return overlay


    def trace_boundingBox(image: np.ndarray,
                          mask: np.ndarray,
                          color: tuple = (0, 0, 255),
                          width: int = 10):
        """
        Draw a bounding box on image

         Parameter
         ----------
         image : image on which we want to draw the box
         mask  : mask to process
         color : color we want to use to draw the box edges
         width : box edges's width

        """

        lbl = label(mask)
        props = regionprops(lbl)
        for prop in props:
            coin1 = (prop.bbox[3], prop.bbox[2])
            coin2 = (prop.bbox[1], prop.bbox[0])
            cv2.rectangle(image, coin2, coin1, color, width)
        return image