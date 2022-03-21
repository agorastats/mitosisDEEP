import cv2
import numpy as np

from maskAnnotations.colorAnnotationsUOCMitosis import CreateMaskAnnotationsUOC
from utils.runnable import Main

mitosis_path = 'data/Laura'
mitosis_images = [str(c).zfill(3) + 'a.jpg' for c in np.arange(21, 41, 1)]


class CreateMaskAnnotationsLaura(CreateMaskAnnotationsUOC):

    def __init__(self):
        super().__init__()
        self.data_path = mitosis_path
        self.images_list = mitosis_images

    def get_annotations(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([10, 93, 0])
        upper = np.array([30, 200, 255])
        mask = cv2.inRange(hsv, lower, upper)
        mask_u8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # keep center of contours with minimum shape
        annot_list = [c.mean(axis=0).astype(int).tolist()[0] for c in contours if c.shape[0] > 40]
        return annot_list


if __name__ == '__main__':
    Main(CreateMaskAnnotationsLaura()).run()
