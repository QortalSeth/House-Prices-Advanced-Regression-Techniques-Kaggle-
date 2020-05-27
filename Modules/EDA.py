import pandas as pd
from Modules import Util as ut

## initialize data
train = pd.read_csv('../Input/train.csv')  # has ids 1-1460, has SalePrice
test = pd.read_csv('../Input/test.csv')    # has ids 1461-2919 doesn't have SalePrice
sampleSubmission = pd.read_csv('../Input/sample_submission.csv') # contains ids and predicted sale prices

columnNotes = pd.read_excel('Column Notes.xlsx')

print(columnNotes['Type'].value_counts())

#categorical columns:
def getColumnType(type: str):
    return columnNotes[columnNotes['Type'] == type]


nominal = train[getColumnType('Nominal')['Name']]
ordinal = train[getColumnType('Ordinal')['Name']]
discrete = train[getColumnType('Discrete')['Name']]

#imputationTypes ={'MSSubClass': ?, 'MSZoning': ?, 'Street': 'Mode', }

## useful aggregate data
nominalCounts = {}
for c in nominal.columns:
    nominalCounts[c] = nominal[c].value_counts()
ut.printDict(nominalCounts, 'Nominal Counts:')

ordinalCounts = {}
for c in ordinal.columns:
    ordinalCounts[c] = ordinal[c].value_counts()
ut.printDict(ordinalCounts, "Ordinal Counts:")

null_columns=train.columns[train.isnull().any()]

nullSums = train[null_columns].isnull().sum().sort_values(ascending=False)
print('Null Columns:')
print(nullSums.to_string)
print('finished EDA')
