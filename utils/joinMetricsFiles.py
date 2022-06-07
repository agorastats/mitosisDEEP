import os
import pandas as pd

from utils.loadAndSaveResults import read_data_frame, store_data_frame

METRICS_PATH = 'data/all_metrics'

if __name__ == '__main__':
    resultsList = [pd.DataFrame()]
    files = [f for f in os.listdir(METRICS_PATH) if f != 'old']
    for file in os.listdir(METRICS_PATH):
        auxDF = read_data_frame(os.path.join(METRICS_PATH, file))
        auxDF.loc[:, 'data'] = pd.Series(file.split('_')[0], index=auxDF.index)
        auxDF.loc[:, 'experiment'] = pd.Series(file, index=auxDF.index)
        resultsList.append(auxDF)

    resultsDF = pd.concat(resultsList, ignore_index=True)
    store_data_frame(resultsDF, os.path.join(METRICS_PATH, 'resultsDF.csv'))

