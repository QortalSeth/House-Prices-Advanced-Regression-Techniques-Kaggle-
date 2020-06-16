from Modules.EDA import *
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

def dfMods(categorical: pd.DataFrame):
    modDF = categorical.copy()
    modDF['GarageScore'] = modDF['GarageQual'] * modDF['GarageArea']
    modDF['TotalFullBath'] = modDF['BsmtFullBath'] + modDF['FullBath']
    modDF['TotalHalfBath'] = modDF['BsmtHalfBath'] + modDF['HalfBath']
    modDF['TotalSF'] = modDF['GrLivArea'] + modDF['TotalBsmtSF']

    if 'SalePrice' in modDF.columns:
        modDF['LogSalePrice'] = np.log(modDF['SalePrice'])

    modDF.drop(columns=['GarageQual', 'GarageArea', 'BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath', 'GrLivArea', 'TotalBsmtSF', 'SalePrice'], inplace=True)
    #modDF[''] = modDF[''] modDF['']

## use vif to reduce muticollinarity
    vifCutoff = 5
    vif = calc_vif(modDF)
    mcColumns = vif[vif['VIF'] > vifCutoff]['variables']
    mcColumns = mcColumns[mcColumns != 'LogSalePrice']
    modDF.drop(columns=mcColumns, inplace=True)
    vif = vif[vif['VIF'] <= vifCutoff]

    print('Finished Modifying Dataframe','\n')
    return modDF

    # average room size column
    # bathroom to room ratio
    # combine porch/deck columns (screened-in, 3Season, OpenPorch, and PoolDeck) into 1 porchSF column
    # remove outliers in numerical data
    # Add variable to determine if house is new based on sale year and built year

