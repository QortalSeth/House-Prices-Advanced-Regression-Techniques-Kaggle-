from sklearn.model_selection import GridSearchCV
import pandas as pd
from Modules.EDA import columnNotes, columnNotes2
from typing import List
from sklearn.preprocessing import StandardScaler

def getColumnType(df, type: str, includeDfMods=False):
    returnValue = columnNotes[columnNotes['Type'] == type]
    if includeDfMods:
        returnValue = returnValue.append(columnNotes2[columnNotes2['Type'] == type])
    returnValue = returnValue[returnValue['Dropped'] != 'Yes']

    return [x for x in returnValue['Name'] if x in df]


def splitDfByCategory(df: pd.DataFrame, includeDfMods=False):

    nominal = df[getColumnType(df, 'Nominal', includeDfMods)].copy()
    ordinal = df[getColumnType(df, 'Ordinal', includeDfMods)].copy()
    discrete = df[getColumnType(df, 'Discrete', includeDfMods)].copy()
    continuous = df[getColumnType(df, 'Continuous', includeDfMods)].copy()
    return nominal, ordinal, discrete, continuous

def scaleData(df: pd.DataFrame, columns: List, inplace=False):
    if inplace:
        result = df
    else:
        result = df.copy()

    sc = StandardScaler()

    for c in columns:
        column = result[[c]]
        column = sc.fit_transform(column)
        column = flatten(column)
        column = pd.Series(column)
        result[c] = column
    if inplace:
        return None
    else:
        return result

flatten = lambda l: [item for sublist in l for item in sublist]