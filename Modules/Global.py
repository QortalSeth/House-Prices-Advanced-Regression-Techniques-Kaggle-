from sklearn.model_selection import GridSearchCV

from Modules.EDA import columnNotes


def getColumnType(type: str):
    returnValue = columnNotes[columnNotes['Type'] == type]
    returnValue = returnValue[returnValue['Dropped'] != 'Yes']
    return returnValue['Name']

