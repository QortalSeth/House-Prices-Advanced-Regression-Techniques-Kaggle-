
#creates dfs from file, prints relevent data
from Modules.EDA import *
import pandas as pd
from Modules.Impute import impute
from Modules.CategoricalEncoding import categoricalEncoding
from Modules.dfMods import dfMods
from Modules.Regressions import performRegressions, predictSalePrice
import Modules.Util as ut
import pickle

## Initialize values
generateModels = True

if generateModels:
## impute data
    imputed = impute(train.copy())

## performs one hot encoding on nominal vars and label encoding on ordinal vars
    categorical = categoricalEncoding(imputed.copy())

## adds or modifies columns to prepare for regressions
    modDF = dfMods(categorical.copy())

## performs regressions and returns dataframe with output
    time, models = ut.getExecutionTime(lambda: performRegressions(modDF))
    ut.pickleObject(models, 'Output/models.pkl')

else:
    models = ut.unpickleObject('Output/models.pkl')
    # Train Size is (1460,81)
    # Imputed Size is (1460,77)
    # Categorical Size is (1460, 234)
    # modDF Size is (1460,231)

    # Test Size is (1459,80)
    # Imputed Size is (1459,76)
    # Categorical Size is (1459, 216) 18 columns missing
    # modDF Size is (1459,216) 15 columns missing



## impute test data
imputedTest = impute(test.copy())

## categorical test data
categoricalTest = categoricalEncoding(imputedTest)
#categoricalDiffColumns = ut.getColumnDiff(categoricalTest, categorical)
#categoricalDiffColumns2 = ut.getColumnDiff(categorical, categoricalTest)

## mod test data
modTest = dfMods(categoricalTest)

# modDfDiffColumns = ut.getColumnDiff(modDF, modTest)
# modDfDiffColumns2 = ut.getColumnDiff(modTest, modDF)

## predict test data
predictSalePrice(modTest, models)


## plot data
print('finished')


