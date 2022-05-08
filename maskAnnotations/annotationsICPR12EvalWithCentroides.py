import pandas as pd

from maskAnnotations.annotationsICPR12Eval import CreateMaskAnnotationsICPR12Eval
from utils.image import create_mask_with_annotations_circle
from utils.runnable import Main


class CreateMaskAnnotationsICPR12EvalWithCentroides(CreateMaskAnnotationsICPR12Eval):

    def __init__(self):
        super().__init__()
        self.mask_output = 'masks_centroid'  # change output of masks folder

    def get_annotations(self, path):
        annot = super().get_annotations(path)
        # shapes to centroides (mean approach) annotations
        return [list(pd.DataFrame(x).mean().astype(int)) for x in annot]

    def create_mask_with_annotations(self, image, annot_list):
        mask = create_mask_with_annotations_circle(image, annot_list)
        return mask


if __name__ == '__main__':
    Main(
        CreateMaskAnnotationsICPR12EvalWithCentroides()
    ).run()
