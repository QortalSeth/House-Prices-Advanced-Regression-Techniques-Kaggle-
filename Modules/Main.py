
#creates dfs from file, prints relevent data
#from Modules.EDA import *

# imputes missing values
from Modules.ImputationTrain import imputed

# performs one hot encoding on nominal vars and label encoding on ordinal vars
from Modules.CategoricalEncoding import categorical

# adds or modifies columns to prepare for regressions
from Modules.dfMods import modDF

# performs regressions
from Modules.Regressions import *

# plots regression data
from Modules.Plots import *

print(len(imputed) + len(categorical)  + len(modDF)  )
print('finished')
