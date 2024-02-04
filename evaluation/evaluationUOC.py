import sys

from evaluation.base import EvaluationMasks
from utils.runnable import Main


DATA_PATH = 'data/UOC Mitosis'
MASK_PATH = 'data/UOC Mitosis/masks'

# OUTPUT_PATH = 'data/Laura/pred_proves_dataset20220207_unet_bce_dice_loss_stain'
# PRED_INFO_CSV = 'pred_proves_dataset20220207_unet_bce_dice_loss_stain.csv'

# --infoCsv pred_proves_dataset20220207_unet_bce_dice_loss_stain.csv'

class EvaluationMasksUOC(EvaluationMasks):

    def __init__(self):
        super().__init__()
        self.data_path = DATA_PATH
        self.mask_path = MASK_PATH
        self.suffix_annot = 'x'

    def run(self, options):
        infoDF = self.get_pred_info(options)
        self.iterate_over_pred_images(infoDF, options)


if __name__ == '__main__':

    argv = sys.argv[1:]
    # argv += ['--infoCsv',
    #          'pred_proves_dataset20220207_unet_bce_dice_loss_stain.csv',
    #          '--output',
    #          'pred_proves_dataset20220207_unet_bce_dice_loss_stain']

    argv += ['--infoCsv',
             'uoc_all_data_2022_03_20_unet_custom_loss_gamma3.csv',
             '--output',
             'uoc_all_data_2022_03_20_unet_custom_loss_gamma3']

    Main(EvaluationMasksUOC()).run(argv)
