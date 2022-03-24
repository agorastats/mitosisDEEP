import os

import cv2

from utils.image import read_image
from utils.loadAndSaveResults import read_data_frame

PATCH_DATA_PATH = 'patches/all_data'

if __name__ == '__main__':

    patchesInfoDF = read_data_frame(os.path.join(PATCH_DATA_PATH, 'infoDF.csv'))
    resultDict = dict()
    for i, item in patchesInfoDF.iterrows():
        img = read_image(os.path.join(PATCH_DATA_PATH, 'masks/%s' % item['id']))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        resultDict[item['id']] = len(contours)
    patchesInfoDF.loc[:, 'mitosisInPatch'] = patchesInfoDF.loc[:, 'id'].map(resultDict)
    pass
