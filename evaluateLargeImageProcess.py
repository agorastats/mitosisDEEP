import logging
import os
import cv2
import numpy as np
import pandas as pd

from patchify import patchify, unpatchify
from utils.image import read_image, create_dir
from utils.loadAndSaveResults import store_data_frame
from utils.runnable import Runnable, Main

# ref:  https://www.kaggle.com/paulorzp/run-length-encode-and-decode
# dice coefficient: https://www.kaggle.com/yerramvarun/understanding-dice-coefficient

def rle_encode(img):
    '''
    # ref: https://www.kaggle.com/stainsby/fast-tested-rle
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


class EvaluateLargeImageProcess(Runnable):
    def __init__(self, df=None, patchify_size=256, img_path=None, mask_path=None, preprocess=None,
                 model=None, output_info=None):
        super().__init__()
        assert img_path is not None, 'need to fill img_path!'
        assert df is not None, 'need to fill df with images info!'
        assert 'id' in df.columns, 'need to fill id column with images info in df'
        assert model is not None
        assert output_info is not None

        self.df = df
        self.patchify_size = patchify_size
        self.img_path = img_path
        self.mask_path = mask_path
        self.preprocess = preprocess
        self.model = model
        self.output_info = output_info
        create_dir(output_info)

    def apply_preprocess(self, img):
        img = img.astype('float64')
        if self.preprocess is None:
            img /= 255.
        else:
            img = self.preprocess(img)
        return img

    def predict_using_patchify(self, img):
        # step same as patch for not overlap patches
        patches = patchify(img, (self.patchify_size, self.patchify_size, 3), step=self.patchify_size)
        patches = patches[:, :, 0, :, :, :]
        predicted_patches = []
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                single_patch = patches[i, j, :, :, :] / 255.  # normalize it
                single_patch = np.expand_dims(single_patch, axis=0)  # (x,y,3) to (1,x,y,3)
                single_patch_prediction = (self.model.predict(single_patch) > 0.5).astype(np.uint8)
                predicted_patches.append(single_patch_prediction[0, :, :])

        predicted_patches_reshaped = np.reshape(predicted_patches,
                                                [patches.shape[0], patches.shape[1], patches.shape[2],
                                                 patches.shape[3]])
        reconstructed_image = unpatchify(predicted_patches_reshaped, (img.shape[0], img.shape[1]))
        return reconstructed_image

    def predict_image(self, img):
        # nearest size divisible by our patch size
        size_x = (img.shape[1] // self.patchify_size) * self.patchify_size
        size_y = (img.shape[0] // self.patchify_size) * self.patchify_size
        logging.info('test image size: (%i, %i)', img.shape[0], img.shape[1])
        img = cv2.resize(img, (size_x, size_y))
        logging.info('test image resized size: (%i, %i)', img.shape[0], img.shape[1])

        img = self.apply_preprocess(img)
        pred_img = self.predict_using_patchify(img)

        return pred_img, size_x, size_y

    def run(self, options):
        infoDFList = list()
        for i, f in enumerate(self.df.loc[:, 'id']):
            logging.info('__predict image %s:' % str(f))
            img = read_image(os.path.join(self.img_path, f))
            name_img = f.split('.')[0]
            pred_img, size_x, size_y = self.predict_image(img)
            # cv2.imwrite(os.path.join(self.output_info, name_img), pred_img)
            auxDF = pd.DataFrame({'id': f, 'size_x': size_x, 'size_y': size_y, 'rle': rle_encode(pred_img)})
            infoDFList.append(auxDF)

        infoDF = pd.concat(infoDFList, ignore_index=True)
        store_data_frame(infoDF, os.path.join(self.output_info, 'pred_info.csv'))


if __name__ == '__main__':
    eval = EvaluateLargeImageProcess(df=pd.DataFrame({'id': ['02x.jpg']}, index=[0]), img_path='sample_data',
                                     model='', output_info='sample_data')
    eval.run({})

    # Main(
    #     EvaluateLargeImageProcess(df=pd.DataFrame({'id': [123]}, index=[0]), img_path='sample_data')
    #     ).run()