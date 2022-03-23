import numpy as np

from maskAnnotations.colorAnnotationsLaura import CreateMaskAnnotationsLaura
from utils.runnable import Main

mitosis_path = 'data/Paula'
mitosis_images = [str(c).zfill(3) + 'x.jpg' for c in np.arange(41, 61, 1)]


class CreateMaskAnnotationsPaula(CreateMaskAnnotationsLaura):

    def __init__(self):
        super().__init__()
        self.data_path = mitosis_path
        self.images_list = mitosis_images

if __name__ == '__main__':
    Main(CreateMaskAnnotationsPaula()).run()
