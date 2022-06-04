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
             'all_data_he_norm_pred_info.csv',
             #'laura_paula_mitosis_only_midog_pred_info.csv',
             '--output',
             'all_data_he_norm_pred_info']

    Main(EvaluationMasksLauraPaula()).run(argv)
