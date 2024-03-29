import os
import numpy as np
import pandas as pd

from tensorflow.keras.utils import Sequence
from utils.augmentation import AUG_IMG_PIPELINE
from utils.image import read_image, read_mask, show_image, show_images_of_generator_item


class DataGenerator(Sequence):
    def __init__(self, df=None, batch_size=16, img_size=256, shuffle=False,
                 preprocess=None, augmentations=AUG_IMG_PIPELINE, img_path=None, mask_path=None, to_predict=False):
        super().__init__()
        assert img_path and mask_path is not None, 'need to fill img_path and mask_path!'
        assert df is not None, 'need to fill df with images, masks info!'
        assert 'id' in df.columns, 'need to fill id column with images info in df'

        self.df = df
        self.batch_size = batch_size
        self.img_size = img_size
        self.n_channels = 3
        self.shuffle = shuffle
        self.preprocess = preprocess
        self.augment = augmentations
        self.img_path = img_path
        self.mask_path = mask_path
        self.to_predict = to_predict
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        """Number of batches in the Sequence."""
        return int(np.floor(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def data_generation(self, indexes):
        X = np.empty((self.batch_size, self.img_size, self.img_size, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size, self.img_size, self.img_size, 1), dtype=np.float32)
        for i, f in enumerate(self.df.loc[:, 'id'].iloc[indexes]):
            X[i, :, :, :] = read_image(os.path.join(self.img_path, f), size=self.img_size)
            if not self.to_predict:
                 y[i, :, :, 0] = read_mask(os.path.join(self.mask_path, f), size=self.img_size)

        if self.preprocess is None:
            X /= 255.
        else:
            X = self.preprocess(X)

        if self.augment:
            img_list, mask_list = [], []
            for x, y in zip(X, y):
                augmented = self.augment(image=x, mask=y)
                img_list.append(augmented['image'])
                mask_list.append(augmented['mask'])
            X = np.array(img_list)
            y = np.array(mask_list)

        return X, y/255.

    def __getitem__(self, index):
        """Gets batches at position index. Using circular logic for indexes"""
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        idxs = self.indexes[start_idx % len(self.indexes):end_idx % len(self.indexes)]
        X, y = self.data_generation(idxs)
        if not self.to_predict:
            return X, y
        else:
            return X


if __name__ == '__main__':
    img_path = 'sample_data/sample_img_patches'
    mask_path = 'sample_data/sample_mask_patches'
    df = pd.DataFrame()
    df.loc[:, 'id'] = os.listdir(img_path)
    gen = DataGenerator(df=df, img_path=img_path, mask_path=mask_path, shuffle=False)
    X, y = gen.__getitem__(0)
    show_image(X[0], y[0])
    # generator with augmentations
    gen = DataGenerator(df=df, img_path=img_path, mask_path=mask_path, shuffle=True, augmentations=AUG_IMG_PIPELINE)
    X, y = gen.__getitem__(1000)
    show_image(X[0], y[0])
    show_images_of_generator_item(X, y)
    pass

