import logging
import os
import cv2
import numpy as np

from maskAnnotations.base import CreateMaskAnnotations, get_annotations_by_zscore
from utils.image import show_image, create_mask_with_annotations_circle
from utils.runnable import Main

UOC_Mitosis_path = 'data/UOC Mitosis'
UOC_Mitosis_images = [str(c).zfill(2) + 'x.jpg' for c in np.arange(1, 36, 1)]


class CreateMaskAnnotationsUOC(CreateMaskAnnotations):

    def __init__(self):
        super().__init__()
        self.data_path = UOC_Mitosis_path
        self.images_list = UOC_Mitosis_images

    def get_annotations(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([170, 25, 0])
        upper = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        mask_u8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # keep center of contours with minimum shape
        annot_list = [c.mean(axis=0).astype(int).tolist()[0] for c in contours if c.shape[0] > 40]
        return annot_list

    def run(self, options):
        for image in self.images_list:
            logging.info('__annotation image: %s' % str(image))
            img = cv2.imread(os.path.join(self.data_path, image))
            if img is not None:
                annot_list = self.get_annotations(img)
                mask = create_mask_with_annotations_circle(img, annot_list)
                # show_image(img, mask)
                cv2.imwrite(os.path.join(self.data_path, self.mask_output, image), mask)


if __name__ == '__main__':
    Main(CreateMaskAnnotationsUOC()).run()
