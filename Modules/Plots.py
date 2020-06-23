from matplotlib import pyplot as plt
import Modules.Util as ut
import matplotlib as mpl
import matplotlib.ticker as mtick

histDir = 'Visualizations/Histograms/'
scatterDir = 'Visualizations/Scatterplots/'

histParams = {'kind': 'hist', 'legend': False, 'bins': 50}
figParams= {'x': 7, 'y': 7}



plt.rc('font', size=40)
plt.rc('axes', labelsize=60)
plt.rc('axes', titlesize=60)

xTickMult = lambda: ut.multiplyRange(plt.xticks()[0], 0.5)
yTickFormat = lambda : plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
xTickFormatPercent = lambda: plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
xTickFormatCommas = lambda: plt.gca().xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
xTickFormatDollars = lambda x=0:  plt.gca().xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('${x:,.'+str(x)+'f}'))
#setTickIn = lambda: plt.gca().tick_params(axis='x', direction='in')
trimTicks = lambda: plt.xticks()[0:-1]

def plotResults(df, models):
    ut.removeOutliers(df[['logSalePrice']]).plot(**histParams)
    ut.multiplyFigSize(**figParams)
    ut.plotSetup({'xlabel' : 'logSalePrice (Dollars)',
                  xTickMult: '',
                  'title'  :'Histogram of Log Sale Price',
                  'grid': None,
                  'savefig': histDir + 'LogSalePrice.png',
                  'show'   : None
                  })

    print('Finished Visualizations')


def plotHists():
    pass

def plotScatter():
    pass



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



# To Do




# plot sq feet to sale price
# plot sale price distribution
# plot missing values
# plot sale price colored by neighborhood if available
# use nullity correlation matrix to see relationship between rows with missing values


