import logging
import os
import cv2

from utils.image import read_image
from utils.loadAndSaveResults import read_data_frame
from utils.runnable import Runnable, Main

PATCH_DATA_PATH = 'patches/all_data'  # to inplace with str data path folder
INFO_DF_NAME = 'infoDF.csv' # assume info of patches located on this csv file


class MitosisInPatchesCount(Runnable):

    def __init__(self):
        super().__init__()
        self.patch_data_path = PATCH_DATA_PATH
        self.info_df = INFO_DF_NAME

    def run(self, options):
        patchesInfoDF = read_data_frame(os.path.join(PATCH_DATA_PATH, self.info_df))
        resultDict = dict()
        for i, item in patchesInfoDF.iterrows():
            if i % 100 == 0:
                logging.info('(progress) Iterating image number: %i / %i' % (i, len(patchesInfoDF)))
            img = read_image(os.path.join(PATCH_DATA_PATH, 'masks/%s' % item['id']))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
            resultDict[item['id']] = len(contours)
            patchesInfoDF.loc[:, 'mitosisInPatch'] = patchesInfoDF.loc[:, 'id'].map(resultDict)
        pass


if __name__ == '__main__':
    Main(
        MitosisInPatchesCount()
    ).run()
