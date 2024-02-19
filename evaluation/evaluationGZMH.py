import sys

from evaluation.base import EvaluationMasks
from utils.runnable import Main

DATA_PATH = 'data/GZMH/test'
MASK_PATH = 'data/GZMH/test/label'


class EvaluationGZMH(EvaluationMasks):

    def __init__(self):
        super().__init__()
        self.data_path = DATA_PATH
        self.mask_path = MASK_PATH

    def get_image_name(self, img):
        return img['id']


if __name__ == '__main__':
    argv = sys.argv[1:]
    argv += ['--infoCsv',
             'pred_info_unet_test_descongelem_encoder_jaccard_2024_gzmh.csv',
             '--output',
             'pred_info_unet_test_descongelem_encoder_jaccard_2024_gzmh']

    Main(EvaluationGZMH()).run(argv)


    # pred_info_unet_test_descongelem_encoder_jaccard_2024_gzmh.csv --> recall:, prec: fscore:    ??