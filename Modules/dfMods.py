from Modules.EDA import *
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

def removeColumnsByVif(df: pd.DataFrame):
    vifCutoff = 5
    #ut.printNulls(modDF)
    #nulls = ut.getNulls(modDF)
    vif = calc_vif(df)
    mcColumns = vif[vif['VIF'] > vifCutoff]['variables']
    mcColumns = mcColumns[mcColumns != 'LogSalePrice']
    df.drop(columns=mcColumns, inplace=True)
    vif = vif[vif['VIF'] <= vifCutoff]
    return vif

def dfMods(categorical: pd.DataFrame):
    categorical['GarageScore'] = categorical['GarageQual'] * categorical['GarageArea']
    categorical['TotalFullBath'] = categorical['BsmtFullBath'] + categorical['FullBath']
    categorical['TotalHalfBath'] = categorical['BsmtHalfBath'] + categorical['HalfBath']
    categorical['TotalSF'] = categorical['GrLivArea'] + categorical['TotalBsmtSF']

    if 'SalePrice' in categorical.columns:
        categorical['LogSalePrice'] = np.log(categorical['SalePrice'])
        categorical.drop(columns=['SalePrice'])

    categorical.drop(columns=['GarageQual', 'GarageArea', 'BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath', 'GrLivArea', 'TotalBsmtSF'], inplace=True)
    #modDF[''] = modDF[''] modDF['']

## use vif to reduce muticollinarity

    #removeColumnsByVif(categorical)
    print('Finished Modifying Dataframe','\n')
    return categorical

    # average room size column
    # bathroom to room ratio
    # combine porch/deck columns (screened-in, 3Season, OpenPorch, and PoolDeck) into 1 porchSF column
    # remove outliers in numerical data
    # Add variable to determine if house is new based on sale year and built year

