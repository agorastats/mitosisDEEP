import osimport numpy as npimport pandas as pdfrom tensorflow.keras.utils import Sequencefrom utils.augmentation import AUGMENTATIONS_PIPELINEfrom utils.image import read_image, read_mask, show_image, show_images_of_generator_itemclass DataGenerator(Sequence):    def __init__(self, df, batch_size=16, img_size=256, n_channels=3, subset="train", shuffle=False,                 preprocess=None, augmentations=None, img_path=None, mask_path=None, to_predict=False):        super(DataGenerator, self).__init__()        self.indexes = None        assert img_path and mask_path is not None, 'need to fill img_path and mask_path!'        assert 'id' in df.columns, 'need to fill id column with images info in df'        self.df = df        self.batch_size = batch_size        self.img_size = img_size        self.n_channels = n_channels        self.subset = subset        self.shuffle = shuffle        self.preprocess = preprocess        self.augment = augmentations        self.img_path = img_path        self.mask_path = mask_path        self.to_predict = to_predict        self.on_epoch_end()    def __len__(self):        """Number of batches in the Sequence."""        return int(np.floor(len(self.df) / self.batch_size))    def on_epoch_end(self):        self.indexes = np.arange(len(self.df))        if self.shuffle:            np.random.shuffle(self.indexes)    def data_generation(self, indexes):        X = np.empty((self.batch_size, self.img_size, self.img_size, self.n_channels), dtype=np.float32)        y = np.empty((self.batch_size, self.img_size, self.img_size, 1), dtype=np.float32)        for i, f in enumerate(self.df.loc[:, 'id'].iloc[indexes]):            X[i, :, :, :] = read_image(os.path.join(self.img_path, f), self.img_size)            if not self.to_predict:                 y[i, :, :, 0] = read_mask(os.path.join(self.mask_path, f), self.img_size)        if self.preprocess is None:            X /= 255.        else:            X = self.preprocess(X)        if self.augment:            img_list, mask_list = [], []            for x, y in zip(X, y):                augmented = self.augment(image=x, mask=y)                img_list.append(augmented['image'])                mask_list.append(augmented['mask'])            X = np.array(img_list)            y = np.array(mask_list)        return X, y/255.    def __getitem__(self, index):        """Gets batches at position index."""        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]        X, y = self.data_generation(indexes)        if not self.to_predict:            return X, y        else:            return Xif __name__ == '__main__':    img_path = 'patches/images_2022_01_12'    mask_path = 'patches/masks_2022_01_12'    df = pd.DataFrame()    df.loc[:, 'id'] = os.listdir(img_path)    gen = DataGenerator(df=df, img_path=img_path, mask_path=mask_path, shuffle=False)    X, y = gen.__getitem__(1)    show_image(X[0], y[0])    # generator with augmentations    gen = DataGenerator(df=df, img_path=img_path, mask_path=mask_path, shuffle=False, augmentations=AUGMENTATIONS_PIPELINE)    X, y = gen.__getitem__(1)    show_image(X[0], y[0])    show_images_of_generator_item(X, y)    pass