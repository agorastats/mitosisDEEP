from os import path, makedirs
import pandas as pd


def store_data_frame(resultsDF, filePath, sep=';', mode='w', header=True, index=False, floatFormat='%.2f'):
    # mode = 'a' append, 'w' write.
    # header = True or False

    folderPath = path.dirname(path.abspath(filePath))
    if not path.exists(folderPath):
        makedirs(folderPath)

    if mode == 'a':
        if not path.exists(filePath):
            headerA = header
        else:
            headerA = False

    if not resultsDF.empty:
        if mode == 'a':
            resultsDF.to_csv(filePath, sep=sep, encoding='utf-8', mode=mode, header=headerA, index=index,
                             float_format=floatFormat)
        else:
            resultsDF.to_csv(filePath, sep=sep, encoding='utf-8', mode=mode, header=header, index=index,
                             float_format=floatFormat)

    else:  # creem el fitxer buit si no hi ha dades
        file = open(filePath, "w")
        file.close()


def read_data_frame(filePath, sep=';', header=0):
    if path.isfile(filePath):
        try:
            resultsDF = pd.read_csv(filePath, header=header, sep=sep)  # , dtype={'key': object})
        except ValueError:
            resultsDF = pd.DataFrame()
    else:
        resultsDF = pd.DataFrame()

    return resultsDF

