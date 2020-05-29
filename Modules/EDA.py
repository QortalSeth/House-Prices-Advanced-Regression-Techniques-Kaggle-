import pandas as pd
from Modules import Util as ut

## initialize data
train = pd.read_csv('Input/train.csv')  # has ids 1-1460, has SalePrice
test = pd.read_csv('Input/test.csv')    # has ids 1461-2919 doesn't have SalePrice
sampleSubmission = pd.read_csv('Input/sample_submission.csv') # contains ids and predicted sale prices

columnNotes = pd.read_excel('Column Notes.xlsx')

# print(columnNotes['Type'].value_counts())

#categorical columns:
def getColumnType(type: str):
    return columnNotes[columnNotes['Type'] == type]


nominal = train[getColumnType('Nominal')['Name']]
ordinal = train[getColumnType('Ordinal')['Name']]
discrete = train[getColumnType('Discrete')['Name']]


print('finished EDA')
