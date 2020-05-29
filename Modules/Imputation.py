from Modules.EDA import *

null_columns=train.columns[train.isnull().any()]

nullSums = train[null_columns].isnull().sum().sort_values(ascending=False)


#imputationTypes ={'MSSubClass': ?, 'MSZoning': ?, 'Street': 'Mode', }
print('Null Columns:')
#print(nullSums)




for c in null_columns:
    print(c, ' ', train[c].isnull().sum())
    print(train[c].value_counts(), '\n')

print(len(null_columns))

print('Finished Imputation')

