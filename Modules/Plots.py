from Modules.dfMods import modDF
from matplotlib import pyplot as plt
import Modules.Util as ut
histDir = 'Visualizations/Histograms/'
scatterDir = 'Visualizations/Scatterplots/'

histParams = {'kind': 'hist', 'legend': False, 'bins': 50}
figParams= {'x': 7, 'y': 7}

plt.rc('font', size=40)
plt.rc('axes', labelsize=60)
plt.rc('axes', titlesize=60)

ut.removeOutliers(modDF[['logSalePrice']]).plot(**histParams)
ut.multiplyFigSize(**figParams)
ut.plotSetup({'xlabel' : 'logSalePrice (Dollars)',
              'xticks'  : ut.multiplyRange(plt.xticks()[0], 0.5),
              'title'  :'Histogram of Log Sale Price',
              'grid': None,
              'savefig': histDir + 'LogSalePrice.png',
              'show'   : None
              })

print('Finished Visualizations')



# To Do




# plot sq feet to sale price
# plot sale price distribution
# plot missing values
# plot sale price colored by neighborhood if available
# use nullity correlation matrix to see relationship between rows with missing values


