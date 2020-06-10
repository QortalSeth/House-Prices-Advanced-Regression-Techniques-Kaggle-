from Modules.EDA import *

# c. data mods


# print(ordinal.columns)



def dfMods(categorical: pd.DataFrame):
    modDF = categorical.copy()
    modDF['GarageScore'] = modDF['GarageQual'] * modDF['GarageArea']
    modDF['TotalFullBath'] = modDF['BsmtFullBath'] + modDF['FullBath']
    modDF['TotalHalfBath'] = modDF['BsmtHalfBath'] + modDF['HalfBath']
    modDF['TotalSF'] = modDF['GrLivArea'] + modDF['TotalBsmtSF']
    modDF['LogSalePrice'] = np.log(modDF['SalePrice'])

    modDF.drop(columns=['GarageQual', 'GarageArea', 'BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath', 'GrLivArea', 'TotalBsmtSF', 'SalePrice'], inplace=True)

    #modDF[''] = modDF[''] modDF['']
    print('Finished Modifying Dataframe','\n')
    return modDF

    # average room size column
    # bathroom to room ratio
    # combine porch/deck columns (screened-in, 3Season, OpenPorch, and PoolDeck) into 1 porchSF column
    # remove outliers in numerical data
    # Add variable to determine if house is new based on sale year and built year

