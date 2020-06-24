import numpy as np

from Modules.Global import splitDfByCategory
from Modules.Impute import *
import Modules.Util as ut

def categoricalEncoding(imputedTrain: pd.DataFrame, imputedTest: pd.DataFrame):

    fullDF = imputedTrain.append(imputedTest)
    nominal, ordinal, discrete, continuous = splitDfByCategory(fullDF)


    # nominal train:    (1460, 23) test: (1459, 23)
    # ordinal train:    (1460, 21) test: (1459, 21)
    # discrete train:   (1460, 14) test: (1459, 14)
    # continuous train: (1460, 17) test: (1459, 16)

    # nominalCounts = {}
    # for c in nominal.columns:
    #     nominalCounts[c] = nominal[c].value_counts()
    # ut.printDict(nominalCounts, 'Nominal Counts:')
    #
    # ordinalCounts = {}
    # for c in ordinal.columns:
    #     ordinalCounts[c] = ordinal[c].value_counts()
    # ut.printDict(ordinalCounts, "Ordinal Counts:")


    qualityDict = {np.nan: 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    bsmtFinType = {np.nan: -1, 'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5}
    ordinal['LotShape'].replace({'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3}, inplace=True)
    ordinal['LandContour'].replace({'Lvl': 0, 'Bnk': 1, 'HLS': 2, 'Low': 3}, inplace=True)
    ordinal['Utilities'].replace({np.nan: -1, 'ELO': 0, 'NoSeWa': 1, 'NoSewr': 2, 'AllPub': 3},inplace=True)
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
    ordinal['GarageFinish'].replace({np.nan: -1, 'Unf': 0, 'RFn': 1, 'Fin': 2}, inplace=True)
    ordinal['GarageQual'].replace(qualityDict, inplace=True)
    ordinal['GarageCond'].replace(qualityDict, inplace=True)
    ordinal['PavedDrive'].replace({'N': 0, 'P': 1, 'Y': 2}, inplace=True)
    #ordinal['PoolQC'].replace(qualityDict, inplace=True)
    ordinal['Fence'].replace({np.nan: -1, 'MnWw': 0, 'GdWo': 1, 'MnPrv': 2, 'GdPrv': 3}, inplace=True)

    #ordinal[''].replace({}, inplace=True)

    #nulls = ut.getNulls(ordinal)
    ordinal = ordinal.applymap(np.int64)
    dummies = pd.DataFrame()

    for c in nominal.columns:
        dummy = pd.get_dummies(nominal[c])
        dummies = pd.concat([dummies, dummy], axis=1)





    categoricalFull = ut.appendColumns([dummies, ordinal, discrete, continuous])

    trainLen = ut.getRowsNum(imputedTrain)

    categoricalTrain = categoricalFull.iloc[0:trainLen,]
    categoricalTest  = categoricalFull.iloc[trainLen:, ].copy()
    categoricalTest.drop(columns= ['SalePrice'], inplace= True)


    #ut.printNulls(categoricalFull)
    #nulls = ut.getNulls(categoricalFull)
    print('Finished Categorical Encoding','\n')
    return categoricalTrain, categoricalTest

