from Modules.EDA import *
import matplotlib.pyplot as plt
null_columns=train.columns[train.isnull().any()]

nullSums = train[null_columns].isnull().sum().sort_values(ascending=False)


#imputationTypes ={'MSSubClass': ?, 'MSZoning': ?, 'Street': 'Mode', }
print('Null Columns:')
#print(nullSums)




for c in null_columns:
    print(c, ' ', train[c].isnull().sum())
    print(train[c].value_counts(), '\n')

print('# of Null columns: ',len(null_columns))

nullsDir = 'Visualizations/Nulls/'
histParams = {'kind': 'hist', 'legend': False, 'bins': 50}

train[['MasVnrArea']].plot(**histParams)
ut.plotSetup({'xlabel' : 'Area in Square Feet',
              #'xticks'  : ut.multiplyRange(plt.xticks()[0], 0.5),
              'title'  :'Histogram of Masonry Veneer Area',
              'grid': None,
              'savefig': nullsDir + 'MasVnrArea.png'
              #'show'   : None
              })

# examine variance of variables. Drop columns where most values are the same. Can be done by dividing most common value count by 2nd most common value count, drop columns below 5%
# (do this before null columns are calculated
# compute missingness of columns by %

print('Finished Imputation')

