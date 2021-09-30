import logging
from utils.reader import read_image

if __name__ == '__main__':
    logging.info('__Test reading images')
    patch_size = (100, 100)
    image = read_image('sample_data/A00_v2/A00_01.bmp')


# per llegir eficient les imatges amb 1 class: https://learnopencv.com/efficient-image-loading/
# referencia base: https://github.com/CODAIT/deep-histopath/blob/master/preprocess_mitoses.py