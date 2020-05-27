from Modules.EDA import *
import numpy as np

qualityDict = {np.nan: -1, 'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
bsmtFinType = {np.nan: -1, 'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5}
ordinal['LotShape'].replace({'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3}, inplace=True)
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
#ordinal[''].replace({}, inplace=True)

#nulls = ordinal.isnull()
ordinal = ordinal.applymap(np.int64)


dummies = pd.DataFrame()

for c in nominal.columns:
    dummy = pd.get_dummies(nominal[c], drop_first=True)
    #dummies = dummies.append(dummy)
    dummies = pd.concat([dummies, dummy], axis=1)

print('finished Categorical Encoding')

categorical = pd.concat([dummies, ordinal], axis=1)