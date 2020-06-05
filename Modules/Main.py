
#creates dfs from file, prints relevent data
#from Modules.EDA import *
import pandas as pd
from Modules.Impute import impute
from Modules.CategoricalEncoding import categoricalEncoding
from Modules.dfMods import dfMods
from Modules.Regressions import performRegressions
## Initialize values
train = pd.read_csv('Input/train.csv')  # has ids 1-1460, has SalePrice
test = pd.read_csv('Input/test.csv')    # has ids 1461-2919 doesn't have SalePrice
sampleSubmission = pd.read_csv('Input/sample_submission.csv') # contains ids and predicted sale prices

## impute data
imputed = impute(train)

## performs one hot encoding on nominal vars and label encoding on ordinal vars
categorical = categoricalEncoding(imputed)

## adds or modifies columns to prepare for regressions
dfMods = dfMods(categorical)

## performs regressions and returns dataframe with output
regressions = performRegressions(dfMods)
# plots regression data

print('finished')


