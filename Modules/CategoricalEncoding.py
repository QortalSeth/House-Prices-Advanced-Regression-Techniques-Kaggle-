import numpy as np

from Modules.Global import splitDfByCategory
from Modules.Impute import *

def categoricalEncoding(imputed: pd.DataFrame):
    nominal, ordinal, discrete, continuous = splitDfByCategory(imputed)

    nominalCounts = {}
    for c in nominal.columns:
        nominalCounts[c] = nominal[c].value_counts()
    #ut.printDict(nominalCounts, 'Nominal Counts:')

    ordinalCounts = {}
    for c in ordinal.columns:
        ordinalCounts[c] = ordinal[c].value_counts()
    #ut.printDict(ordinalCounts, "Ordinal Counts:")


    qualityDict = {np.nan: 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    bsmtFinType = {np.nan: -1, 'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5}
    ordinal['LotShape'].replace({'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3}, inplace=True)
    ordinal['LandContour'].replace({'Lvl': 0, 'Bnk': 1, 'HLS': 2, 'Low': 3}, inplace=True)
    ordinal['Utilities'].replace({'ELO': 0, 'NoSeWa': 1, 'NoSewr': 2, 'AllPub': 3},inplace=True)
    ordinal['LandSlope'].replace({'Gtl': 0, 'Mod': 1, 'Sev': 2}, inplace=True)
    ordinal['ExterQual'].replace(qualityDict, inplace=True)
    ordinal['ExterCond'].replace(qualityDict, inplace=True)

    ordinal['BsmtQual'].replace(qualityDict, inplace=True)
    ordinal['BsmtCond'].replace(qualityDict, inplace=True)
    ordinal['BsmtExposure'].replace({np.nan: -1, 'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3}, inplace=True)
    ordinal['BsmtFinType1'].replace(bsmtFinType, inplace=True)
    ordinal['BsmtFinType2'].replace(bsmtFinType, inplace=True)
    ordinal['HeatingQC'].replace(qualityDict, inplace=True)
    ordinal['KitchenQual'].replace(qualityDict, inplace=True)
    ordinal['FireplaceQu'].replace(qualityDict, inplace=True)
    ordinal['GarageFinish'].replace({'NA': -1, 'Unf': 0, 'RFn': 1, 'Fin': 2}, inplace=True)
    ordinal['GarageQual'].replace(qualityDict, inplace=True)
    ordinal['GarageCond'].replace(qualityDict, inplace=True)
    ordinal['PavedDrive'].replace({'N': 0, 'P': 1, 'Y': 2}, inplace=True)
    #ordinal['PoolQC'].replace(qualityDict, inplace=True)
    ordinal['Fence'].replace({'NA': -1, 'MnWw': 0, 'GdWo': 1, 'MnPrv': 2, 'GdPrv': 3}, inplace=True)

    #ordinal[''].replace({}, inplace=True)

    ordinal = ordinal.applymap(np.int64)
    dummies = pd.DataFrame()

    for c in nominal.columns:
        dummy = pd.get_dummies(nominal[c], drop_first=True)
        dummies = pd.concat([dummies, dummy], axis=1)





    categorical = pd.concat([ordinal, discrete.copy(), continuous.copy()], axis=1)
    print('finished Categorical Encoding','\n')
    return categorical, dummies

