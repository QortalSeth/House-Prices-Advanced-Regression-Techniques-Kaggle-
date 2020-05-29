from Modules.EDA import *
from Modules.CategoricalEncoding import categorical

# c. data mods


# print(ordinal.columns)

modDF = categorical.copy()
modDF['GarageScore'] = modDF['GarageQual'] * modDF['GarageArea']
modDF['TotalFullBath'] = modDF['BsmtFullBath'] + modDF['FullBath']
modDF['TotalHalfBath'] = modDF['BsmtHalfBath'] + modDF['HalfBath']
modDF['TotalSF'] = modDF['GrLivArea'] + modDF['TotalBsmtSF']
modDF['logSalePrice'] = np.log(modDF['SalePrice'])

#modDF[''] = modDF[''] modDF['']
print('Finished Modifying Dataframe')


# average room size column
# bathroom to room ratio
# combine porch/deck columns (screened-in, 3Season, OpenPorch, and PoolDeck) into 1 porchSF column


# remove outliers in numerical data





# Add variable to determine if house is new based on sale year and built year
#
