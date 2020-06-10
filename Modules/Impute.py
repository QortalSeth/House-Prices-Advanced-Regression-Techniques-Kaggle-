from Modules.EDA import *
import math

from Modules.Global import getColumnType

def impute(df):
    imputed = train.copy()

    # ut.printNulls(imputed)

    nullsDir = 'Visualizations/Nulls/'
    histParams = {'kind': 'hist', 'legend': False, 'bins': 100}

# train[['MasVnrArea']].plot(**histParams)
# ut.plotSetup({'xlabel' : 'Area in Square Feet',
#               #'xticks'  : ut.multiplyRange(plt.xticks()[0], 0.5),
#               'title'  :'Histogram of Masonry Veneer Area',
#               'grid': None,
#               'savefig': nullsDir + 'MasVnrArea.png'
#               #'show'   : None
#               })
#
# train[['LotFront'
#        'age']].plot(**histParams)
# ut.plotSetup({'xlabel' : 'Length in Feet',
#               #'xticks'  : ut.multiplyRange(plt.xticks()[0], 0.5),
#               'title'  :'Histogram of Lot Frontage Area',
#               'grid': None,
#               'savefig': nullsDir + 'LotFrontageArea.png'
#               #'show'   : None
#               })

# examine variance of variables. Drop columns where most values are the same. Can be done by dividing most common value count by 2nd most common value count, drop columns below 5%
# (do this before null columns are calculated
# compute missingness of columns by %

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
    imputed['GarageFinish'].fillna('NA', inplace=True)
    imputed['Fence'].fillna('NA', inplace=True)

#BsmtQual, Cond, Exposure, FinType1, FinType2 no imputation needed
# GarageQual, GarageCond not needed

# Mode Imputation:
    imputed['Electrical'].fillna(imputed['Electrical'].mode()[0], inplace=True)



#Drop column due to too much missingness
    imputed.drop(columns=['PoolQC', 'PoolArea', 'MiscFeature', 'MiscVal'], inplace=True, axis=1)
#ut.printNulls(imputed)
    nulls = imputed[imputed['Electrical'].isnull()]

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
