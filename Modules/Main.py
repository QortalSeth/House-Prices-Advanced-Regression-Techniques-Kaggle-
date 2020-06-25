
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
featureSelectVIF = False


## impute data
imputed = impute(train.copy())
imputedTest = impute(test.copy())

## performs one hot encoding on nominal vars and label encoding on ordinal vars
categorical, categoricalTest = categoricalEncoding(imputed.copy(), imputedTest.copy())


if generateModels:
## adds or modifies columns to prepare for regressions
    modDF = dfMods(categorical.copy(), featureSelectVIF)

## performs regressions and returns dataframe with output
    time, models = ut.getExecutionTime(lambda: performRegressions(modDF))
    ut.pickleObject(models, 'Output/models.pkl')

else:
    models = ut.unpickleObject('Output/models.pkl')
    # Train Size is (1460,81)
    # Imputed Size is (1460,77)
    # Categorical Size is (1460, 235)
    # modDF Size is (1460,231)

    # Test Size is (1459,80)
    # Imputed Size is (1459,76)
    # Categorical Size is (1459, 230) 5 columns missing
    # modDF Size is (1459,230)


## mod test data
modTest = dfMods(categoricalTest, featureSelectVIF)

# modDfDiffColumns = ut.getColumnDiff(modDF, modTest)
# modDfDiffColumns2 = ut.getColumnDiff(modTest, modDF)

## predict test data
predictSalePrice(modTest, models)


## plot data
print('finished')


