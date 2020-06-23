
import math
import pandas as pd

from Modules.Global import getColumnType

def impute(imputed: pd.DataFrame):
    MSDict = {20: '1 Story After 1946', 30: '1 Story before 1946', 40: '1 Story With Attic',
           45: '1.5 Story Unfinished', 50: '1.5 Story Finished', 60: '2 Story after 1946',
           70: '2 Story Before 1946', 75: '2.5 Story', 80: 'Split or Multi-level',
           85: 'Split Foyer', 90: 'Duplex', 120: '1 Story PUD', 150: '1.5 Story PUD',
           160: '2 Story PUD', 180: 'Mutilevel PUD', 190: '2 Family Conversion'}

    imputed['MSSubClass'].replace(MSDict, inplace=True)

   # ut.printNulls(imputed)

# examine variance of variables. Drop columns where most values are the same. Can be done by dividing most common value count by 2nd most common value count, drop columns below 5%
# (do this before null columns are calculated

## Imputation Section

# Lot Frontage
# impute average of neighborhood
    nh = imputed.groupby(by='Neighborhood').mean()[['LotFrontage']].to_dict()['LotFrontage']
    imputed['LotFrontage'] = imputed.apply(lambda row: nh[row['Neighborhood']] if math.isnan(row['LotFrontage']) else row['LotFrontage'], axis=1 )

# Garage Year Built is same as house year built if null
    imputed['GarageYrBlt'] = imputed.apply(lambda row: row['YearBuilt'] if math.isnan(row['GarageYrBlt']) else row['GarageYrBlt'], axis=1)

#Impute NA as class:
    imputed['Alley'].fillna('No Alley', inplace=True)
    imputed['MasVnrType'].fillna('None', inplace=True)
    imputed['MasVnrArea'].fillna(0, inplace=True)
    imputed['GarageType'].fillna('NA', inplace=True)

    #imputed['GarageYrBlt'].fillna('NA', inplace=True)
 #   imputed['GarageFinish'].fillna('NA', inplace=True)
 #   imputed['Fence'].fillna('NA', inplace=True)

#BsmtQual, Cond, Exposure, FinType1, FinType2 no imputation needed
# GarageQual, GarageCond not needed

# Mode Imputation:
    imputed['Electrical'].fillna(imputed['Electrical'].mode()[0], inplace=True)
    imputed['GarageCars'].fillna(imputed['GarageCars'].mode()[0], inplace=True)
    imputed['BsmtFinSF1'].fillna(imputed['BsmtFinSF1'].mode()[0], inplace=True)
    imputed['BsmtFinSF2'].fillna(imputed['BsmtFinSF2'].mode()[0], inplace=True)
    imputed['BsmtUnfSF'].fillna(imputed['BsmtUnfSF'].mode()[0], inplace=True)
    imputed['BsmtFullBath'].fillna(imputed['BsmtFullBath'].mode()[0], inplace=True)
    imputed['BsmtHalfBath'].fillna(imputed['BsmtHalfBath'].mode()[0], inplace=True)
    imputed['TotalBsmtSF'].fillna(imputed['TotalBsmtSF'].mode()[0], inplace=True)
    imputed['GarageArea'].fillna(imputed['GarageArea'].mode()[0], inplace=True)
    # imputed[''].fillna(imputed[''].mode()[0], inplace=True)


#Drop column due to too much missingness
    imputed.drop(columns=['PoolQC', 'PoolArea', 'MiscFeature', 'MiscVal'], inplace=True, axis=1)


    print('Finished Imputation','\n')
    return imputed


# Fill NAs with Mode
#     Electrical
#     MSZoning
#     Utilities
#     Exterior1st
#     Exterior2nd
#     SaleType
#
# Fill NAs with 0
#
#     MasVnrArea
#     LotFrontage
#     BsmtFullBath
#     BsmtHalfBath
#
# Fill NAs with ‘None’
#
#     MasVnrType
#     PoolQC
#     Fence
#     MiscFeature
#     GarageType
