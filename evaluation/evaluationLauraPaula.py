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
             'lauraPaula_pred_info_proves_amb_midog_unet_2022_05_05.csv',
             #'laura_paula_mitosis_only_midog_pred_info.csv',
             '--output',
             'lauraPaula_pred_info_proves_amb_midog_unet_2022_05_05']

    Main(EvaluationMasksLauraPaula()).run(argv)
