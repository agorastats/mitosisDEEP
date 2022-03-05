import cv2
import numpy as np
from utils.image import show_image, read_image
from utils.loadAndSaveResults import read_data_frame

if __name__ == '__main__':

    # https://stackoverflow.com/questions/65138694/opencv-blob-defect-anomaly-detection
    img = cv2.imread("sample_data/03x.jpg")
    show_image(img)
    mean = np.mean(img, axis=(0, 1))
    std = np.std(img, axis=(0, 1))
    mask = (np.abs(img - mean) / std >= 4.5).any(axis=2)
    mask_u8 = mask.astype(np.uint8) * 255
    show_image(mask_u8)
    # todo: si pasem a thresh binari funcionara millor !
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = sorted(contours, key=cv2.contourArea, reverse=True)
    # cv2.drawContours(img, [each candidates coords], -1, (255, 0, 0), 3)
    pass

    patchesInfoDF = read_data_frame('patches/20220207/infoDF.csv')

    resultDict = dict()
    for i, item in patchesInfoDF.iterrows():
        img = read_image('patches/20220207/masks/%s' % item['id'])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        resultDict[item['id']] = len(contours)
    patchesInfoDF.loc[:, 'mitosisInPatch'] = patchesInfoDF.loc[:, 'id'].map(resultDict)
    pass
