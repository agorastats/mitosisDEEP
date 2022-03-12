import json
import logging
from os import path, makedirs
import pandas as pd
from bson import json_util


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


def read_json(filePath, returnError=False):
    jsonContent = {}  # jsonContent can be a DICTIONARY or a LIST or ...

    errorReading = False
    if path.exists(filePath):
        try:
            with open(filePath) as fp:
                jsonContent = json.load(fp, object_hook=json_util.object_hook)
        except:
            errorReading = True
            logging.error(' Error reading json ' + filePath + '\n\n', exc_info=True)
    else:
        errorReading = True
        logging.warning('File ' + filePath + ' does not exist')

    if returnError == True:
        return jsonContent, errorReading
    else:
        return jsonContent  # jsonContent can be a DICTIONARY or a LIST or ...
