import logging
import os
import time

import cv2
import numpy as np
import pandas as pd

from patchify import patchify, unpatchify
from utils.image import read_image, create_dir, rle_encode
from utils.loadAndSaveResults import store_data_frame
from utils.runnable import Runnable
from utils.stain.preprocessStain import Normalizer

DEFAULT_STAIN_REF_IMG = 'utils/stain/he_ref/A00_01_ref_img.bmp'


class EvaluateLargeImageProcess(Runnable):
    def __init__(self, df=None, patchify_size=256, overlap_patches=1, img_path=None, mask_path=None,
                 preprocess=None, cutoff=0.5,
                 model=None, stain=False, stain_ref_img=None, output_info=None):
        super().__init__()
        assert img_path is not None, 'need to fill img_path!'
        assert df is not None, 'need to fill df with images info!'
        assert (patchify_size % overlap_patches) == 0, 'problems with patchify size and overlap on patches. ' \
                                                       'Patch size / overlap module must be 0! '
        assert 'id' in df.columns, 'need to fill id column with images info in df'
        assert model is not None
        assert output_info is not None

        self.df = df
        self.patchify_size = patchify_size
        self.overlap_patches = overlap_patches
        self.patchify_step = int(self.patchify_size / self.overlap_patches)
        self.img_path = img_path
        self.mask_path = mask_path
        self.preprocess = preprocess
        self.model = model
        self.output_info = output_info
        create_dir(output_info)

        # resolve stain
        self.stain_norm = None
        self.stain = [False] if stain is False else [False, True]
        if any(self.stain):
            self.init_stain_norm(DEFAULT_STAIN_REF_IMG if stain_ref_img is None else stain_ref_img)

        self.cutoff = cutoff

    def init_stain_norm(self, img):
        img = read_image(img)
        self.stain_norm = Normalizer()
        self.stain_norm.fit(img)

    def apply_stain_normalization(self, img):
        img = self.stain_norm.transform(img)
        return img

    def apply_preprocess(self, img, stain):
        img_preproc = img.copy()
        img_preproc = img_preproc.astype('float32')

        if stain:
            img_preproc = self.apply_stain_normalization(img_preproc)

        if self.preprocess is None:
            img_preproc = img_preproc / 255.
        else:
            img_preproc = self.preprocess(img_preproc)
        return img_preproc

    def predict_using_patchify(self, img, stain):
        # step same as patch for not overlap patches
        img_preproc = self.apply_preprocess(img, stain)
        patches = patchify(img_preproc, (self.patchify_size, self.patchify_size, 3), step=self.patchify_step)
        patches = patches[:, :, 0, :, :, :]
        predicted_patches = []
        logging.info('____predict patches')
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                single_patch = patches[i, j, :, :, :]
                single_patch = np.expand_dims(single_patch, axis=0)  # (x,y,3) to (1,x,y,3)
                single_patch_prediction = (self.model.predict(single_patch) >= self.cutoff).astype(np.uint8) 
                # new approach applying treatment of initial prediction
                # morphological transformation over patch
                kernel5 = np.ones((5, 5), np.uint8)  # to use in cv2 methods
                redefined_mask = cv2.dilate(single_patch_prediction[0:, :, ] * 255., kernel5, iterations=1)
                redefined_mask = cv2.erode(redefined_mask, kernel5, iterations=1)
                redefined_mask = cv2.morphologyEx(redefined_mask, cv2.MORPH_CLOSE, kernel5)
                # get contours
                contours, _ = cv2.findContours(redefined_mask[0, :, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                new_redefined_patch = np.zeros_like(redefined_mask)
                for contour in contours:
                    # Calcular la probabilidad promedio de los píxeles dentro del contorno
                    average_probability = np.mean(redefined_mask[0, contour[:, 1], contour[:, 0]])
                    # Calcular el área del contorno
                    area = cv2.contourArea(contour)
                    # Filtrar contornos basados en probabilidad y otras métricas de forma
                    if average_probability >= 0.7 and area >= 10:
                        # Rellenar el contorno en la imagen en blanco con un valor específico (por ejemplo, 1)
                        cv2.drawContours(new_redefined_patch, [contour], -1, 1, thickness=cv2.FILLED)

                predicted_patches.append(new_redefined_patch.astype(np.uint8))

        predicted_patches_reshaped = np.reshape(predicted_patches,
                                                [patches.shape[0], patches.shape[1], patches.shape[2],
                                                 patches.shape[3]])
        logging.info('____reconstruct image with patches')
        reconstructed_image = unpatchify(predicted_patches_reshaped, (img_preproc.shape[0], img_preproc.shape[1]))
        return reconstructed_image

    def predict_image(self, img, stain=False):
        logging.info('__predict with stain: %s' % str(stain))
        # nearest size divisible by our patch size
        size_x = (img.shape[1] // self.patchify_size) * self.patchify_size
        size_y = (img.shape[0] // self.patchify_size) * self.patchify_size
        logging.info('___test image size: (%i, %i)', img.shape[1], img.shape[0])
        img = cv2.resize(img, (size_x, size_y))
        logging.info('___test image resized size: (%i, %i)', img.shape[1], img.shape[0])
        pred_img = self.predict_using_patchify(img, stain)

        return pred_img, size_x, size_y

    def run(self, options):
        infoDFList = list()
        for i, f in enumerate(self.df.loc[:, 'id']):
            init_time = time.time()
            logging.info('predict image: %s' % str(f))
            img = read_image(os.path.join(self.img_path, f))
            for stain in self.stain:
                pred_img, size_x, size_y = self.predict_image(img, stain=stain)
                values_dict = {'id': f, 'stain': stain, 'size_x': size_x, 'size_y': size_y, 'rle': rle_encode(pred_img)}
                auxDF = pd.DataFrame(values_dict, index=[0])
                infoDFList.append(auxDF)
            logging.info('__total time for prediction: ' + str(round(time.time() - init_time, 2)) + ' seconds')

        infoDF = pd.concat(infoDFList, ignore_index=True)
        store_data_frame(infoDF, os.path.join(self.output_info, 'pred_info.csv'))
