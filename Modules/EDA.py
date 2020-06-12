import pandas as pd
import numpy as np
from Modules import Util as ut

## initialize data
train = pd.read_csv('Input/train.csv')  # has ids 1-1460, has SalePrice
test = pd.read_csv('Input/test.csv')    # has ids 1461-2919 doesn't have SalePrice
sampleSubmission = pd.read_csv('Input/sample_submission.csv') # contains ids and predicted sale prices

columnNotes = pd.read_excel('Column Notes.xlsx', sheet_name='Original Columns')
columnNotes2 = pd.read_excel('Column Notes.xlsx', sheet_name='Additional Columns')
# print(columnNotes['Type'].value_counts())

MSDict = {20: '1 Story After 1946', 30: '1 Story before 1946', 40: '1 Story With Attic',
     45: '1.5 Story Unfinished', 50: '1.5 Story Finished', 60: '2 Story after 1946',
     70: '2 Story Before 1946', 75: '2.5 Story', 80: 'Split or Multi-level',
     85: 'Split Foyer', 90: 'Duplex', 120: '1 Story PUD', 150: '1.5 Story PUD',
     160: '2 Story PUD', 180: 'Mutilevel PUD', 190: '2 Family Conversion'}

train['MSSubClass'].replace(MSDict, inplace=True)

print('finished EDA','\n')
