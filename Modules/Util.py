import pandas as pd
import datetime as dt
from typing import List, Dict, Callable
import matplotlib.pyplot as plt


def getRows(df: pd.DataFrame) -> int:
    return len(df.index)


def getColumns(df: pd.DataFrame) -> int:
    return len(df.columns)


def printList(list: List, start=''):
    print(start)
    for i in list:
        print(i)
    print('')


def printDict(dict: Dict, start=''):
    print(start)
    print(type(dict))
    for k, v in dict.items():
        print(k)
        print(v,'\n')

def dictDiff(dict1: Dict[str,int], dict2: Dict[str,int]):
    result = {}
    for i in dict1.keys():
        result[i] = dict1[i] - dict2[i]
    return result


def dfTypes(df: pd.DataFrame):
    typesDict = {}
    for c in df.columns:
        typesDict[c] = df[c].apply(type).value_counts()
    return typesDict

def seriesTypes(s: pd.Series):
    return s.apply(type).value_counts()


def dateFormat(df: pd.DataFrame, dateFormat='%m/%d/%Y'):
    dateColumns = []

    for k, v in dfTypes(df).items():
        if dt.datetime in v or pd.Timestamp in v or dt.time in v:
            dateColumns.append(k)

    for column in dateColumns:
        types = seriesTypes(df[column])
        df[column] = df[column].apply(lambda date: pd.to_datetime(date.strftime(dateFormat), infer_datetime_format=True).date() if isinstance(date, dt.datetime) and date is not pd.NaT else None)

        # print(types)

    for column in df.columns:
        if isinstance(column, dt.datetime):
            df.rename(columns={column: column.date().strftime(dateFormat)}, inplace=True)
            # print(df.columns)


def removeEmptyAll(list: List[pd.DataFrame]):
    for df in list:
        removeEmpty(df)

def removeEmpty(df: pd.DataFrame, printEmpty=False):
    rowCountBefore = getRows(df)
    columnCountBefore = getColumns(df)

    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    rowsRemoved = rowCountBefore - getRows(df)
    columnsRemoved = columnCountBefore - getColumns(df)

    if(printEmpty):
        print(rowsRemoved, ' of ', rowCountBefore, ' rows removed.')
        print(columnsRemoved, ' of ', columnCountBefore, ' columns removed.\n')


def validateNum(s: pd.Series, type='positive', text='', fun=lambda x: x):
    if type is 'positive':
        fun = lambda x : x > 0

    elif type is 'negative':
        fun = lambda x: x < 0

    validInts = len(s[fun(s)])
    totalInts = len(s)
    print(text, str(totalInts - validInts),' of ', str(totalInts), ' numbers are invalid.')

    return s[s[fun(s) == False]]

def printSum(s: pd.Series, text='', sig=2):
    seriesSum = s.sum()
    sigString = '%.'+str(sig)+'f'
    print(text,  sigString %(seriesSum))

def printMean(s: pd.Series, text='', sig=2):
    seriesSum = s.mean()
    sigString = '%.'+str(sig)+'f'
    print(text,  sigString %(seriesSum))





def validateType(s: pd.Series, t, text=''):
    validTypeCount = s.apply(type).value_counts().to_dict()[t]
    totalTypes = len(s)
    print(text, str(totalTypes - validTypeCount), ' of ', str(totalTypes), ' types are invalid.')

def colorByPositivity(df, column, output: Callable):
    posDF = df[df[column] >= 0]
    negDF = df[df[column] >= 0]

def roundFloats(df: pd.DataFrame):
    return df.applymap(lambda x: roundTraditional(x, 2) if isinstance(x, float) else x)

def roundTraditional(number, ndigits):
    return round(number + 10 ** (-len(str(number)) - 1), ndigits)

def setFloatPrecision(df: pd.DataFrame):
    df = df.applymap("${0:.2f}".format)

def multiplyPercentBy100 (df: pd.DataFrame):
    columns = []
    for c in df.columns:
        if '%' in c:
            columns.append(c)

    for c in columns:
        df[c] = df[c].apply(lambda x: x*100)

    return df

def convertColumns(df: pd.DataFrame, oldType, newType):
    columnsToConvert = df.select_dtypes(include=[oldType])

    for col in columnsToConvert.columns.values:
        df[col] = df[col].astype(newType)




def plotSetup(**params):

    for f,p in params.items():
        getattr(plt, f)(p)


