
#creates dfs from file, prints relevent data
from Modules.EDA import *
import pandas as pd
from Modules.Impute import impute
from Modules.CategoricalEncoding import categoricalEncoding
from Modules.dfMods import dfMods
from Modules.Regressions import performRegressions, predictSalePrice
import Modules.Util as ut
import Modules.Plots as plots
import pickle

## Initialize values
generateModels = False
featureSelectVIF = False

fullDF = train.append(test)

## impute data
imputed = impute(fullDF.copy())

## performs one hot encoding on nominal vars and label encoding on ordinal vars
categorical = categoricalEncoding(imputed.copy())

## adds or manipulates columns
modDF = dfMods(categorical.copy(), featureSelectVIF)

## adds or modifies columns to prepare for regressions

trainLen = ut.getRowsNum(train)
modDfTrain = modDF.iloc[0:trainLen, ]
modDfTest = modDF.iloc[trainLen:, ].copy()
modDfTest.drop(columns=['LogSalePrice'], inplace=True)

## performs regressions and returns dataframe with output
if generateModels:

    time, models = ut.getExecutionTime(lambda: performRegressions(modDfTrain))
    ut.pickleObject(models, 'Output/models.pkl')
else:
    models = ut.unpickleObject('Output/models.pkl')



# modDfDiffColumns = ut.getColumnDiff(modDF, modTest)
# modDfDiffColumns2 = ut.getColumnDiff(modTest, modDF)

## plot data
plots.plotResults(train,modDfTrain, models)

## predict test data
predictSalePrice(modDfTest, models)


## plot data
print('Finished')


