import sys

from evaluation.evaluationUOC import EvaluationMasksUOC
from utils.runnable import Main

DATA_PATH = 'data/LauraPaula'
MASK_PATH = 'data/LauraPaula/masks'


class EvaluationMasksLauraPaula(EvaluationMasksUOC):

    def __init__(self):
        super().__init__()
        self.data_path = DATA_PATH
        self.mask_path = MASK_PATH
        self.suffix_annot = 'a'


if __name__ == '__main__':
    argv = sys.argv[1:]
    argv += ['--infoCsv',
             'laura_paula_all_data_2022_03_20_unet_custom_loss.csv',
             '--output',
             'laura_paula_all_data_2022_03_20_unet_custom_loss']

    Main(EvaluationMasksLauraPaula()).run(argv)

# laura_paula_all_data_2022_03_20_unet_custom_loss.csv: recall 0.691, prec 0.611, fscore: 0.611
# lauraPaula_pred_info_all_data_2022_03_20_unet: recall 0.733, prec: 0.62, fscore: 0.63
# lauraPaula_pred_info_all_data_2022_03_20_resnet: recall 0.72, prec: 0.818, fscore: 0.72