from Modules.EDA import *

# c. data mods

# use label encoding on ordinal vars to transform them into integers

print(ordinal.columns)

train['newGarageScore'] = 5

# columns to add:
# create garage interaction column = garage quality * # of cars
# create total Full Bath/ total half bath columns
# average room size column
# bathroom to room ratio
# combine porch/deck columns (screened-in, 3Season, OpenPorch, and PoolDeck) into 1 porchSF column
# add total square foot column using 'GrLivArea' + 'TotalBsmtSF'


# remove outliers in numerical data
# normalize sale price distribution by creating logprice column
# use median, mode, or random imputation on missing values. perhaps try different imputation methods and see how they influence score
# compute missingness of columns by %

# examine variance of variables. Drop columns where most values are the same. Can be done by dividing most common value count by 2nd most common value count, drop columns below 5%
# Add variable to determine if house is new based on sale year and built year
#
