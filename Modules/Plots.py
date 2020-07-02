from matplotlib import pyplot as plt
import Modules.Util as ut
import matplotlib as mpl
import matplotlib.ticker as mtick
import pandas as pd

histDir = 'Output/Histograms/'
scatterDir = 'Output/Scatterplots/'
barDir = 'Output/Barplots/'

histParams = {'kind': 'hist', 'legend': False, 'bins': 50}
barParams = {'kind': 'bar', 'legend': False}
figParams= {'x': 7, 'y': 7}



plt.rc('font', size=40)
plt.rc('axes', labelsize=60)
plt.rc('axes', titlesize=60)

xTickMult = lambda: ut.multiplyRange(plt.xticks()[0], 0.5)
xTickMultLS = lambda: ut.multiplyLinSpace(plt.xticks()[0], 2)
yTickFormat = lambda : plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
xTickFormatPercent = lambda: plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
xTickFormatCommas = lambda: plt.gca().xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
xTickFormatDollars = lambda x=0:  plt.gca().xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('${x:,.'+str(x)+'f}'))
#setTickIn = lambda: plt.gca().tick_params(axis='x', direction='in')
trimTicks = lambda: plt.xticks()[0:-1]
nullsDir = 'Visualizations/Nulls/'
histParams = {'kind': 'hist', 'legend': False, 'bins': 100}

def plotResults(train, modDfTrain, models, ):
    ut.plotDF(train[['SalePrice']], histParams,
           {xTickFormatDollars: '',
            yTickFormat: '',
            'grid': None,
            'xlabel': 'Sale Price',
            'title': 'Histogram of Sale Price of Houses in Ames, Iowa',
            'savefig': histDir + 'SalePrice.png'})

    ut.plotDF(modDfTrain[['LogSalePrice']], histParams,
           {
            yTickFormat: '',
            'grid': None,
            'xlabel': 'Log Sale Price',
            'title': 'Histogram of Log Sale Price of Houses in Ames, Iowa',
            'savefig': histDir + 'LogSalePrice.png'})


    nulls = ut.getNullColumns(train)
    nullsP = ut.getNullPercents(train)


    ut.plotDF(nullsP, {'kind': 'barh', 'x': 'Column','y': 'Null Percent', 'legend': False },
           {
            'grid': None,
            xTickFormatPercent: '',
            'xlabel': '# of Null Values',
            'title': 'Bar Plot of Null Columns ',
            'savefig': barDir + 'Null Percents.png'}, removeOutliersBeforePlotting=False)

    modelResults = pd.read_excel('Output/Model Results.xlsx')

    ut.plotDF(modelResults.sort_values(by='testScore', ascending=True), {'kind': 'barh', 'x': 'name','y': 'testScore', 'legend': False },
           {
            'xticks': xTickMultLS,
            'grid': None,
            'xlabel': 'Test Score',
            'ylabel': 'Model Name',
            'title': 'Bar Plot of Model Scores ',
            'savefig': barDir + 'Model Scores.png'}, removeOutliersBeforePlotting=False)


    modelResults['time'] = modelResults['time'].apply(ut.getTime)
    ut.plotDF(modelResults.sort_values(by='time', ascending=True), {'kind': 'barh', 'x': 'name','y': 'time', 'legend': False },
           {
            'xticks': xTickMult,
            'grid': None,
            'xlabel': 'Time (Seconds)',
            'ylabel': 'Model Name',
            'title': 'Bar Plot of Model Times ',
            'savefig': barDir + 'Model Times.png'}, removeOutliersBeforePlotting=False)

    lasso = models['Lasso']
    coefs = lasso.model.coef_
    columns = modDfTrain.drop(columns='LogSalePrice').columns

    lassoCoefs = pd.DataFrame({'Variable': columns, 'Coefficient': coefs})
    lassoCoefs = lassoCoefs[lassoCoefs['Coefficient'] > 0.001].sort_values(by='Coefficient', ascending=True)


    ut.plotDF(lassoCoefs, {'kind': 'barh', 'x': 'Variable','y': 'Coefficient', 'legend': False },
           {
            'xticks': xTickMultLS,
            'grid': None,
            'xlabel': 'Lasso Coefficient',
            'ylabel': 'Variable Name',
            'title': 'Bar Plot of Lasso Feature Selection ',
            'savefig': barDir + 'Lasso Feature Selection.png'}, removeOutliersBeforePlotting=False)


    gbm = models['Gradient Boost']
    coefs = gbm.model.feature_importances_

    gbmCoefs = pd.DataFrame({'Variable': columns, 'Coefficient': coefs}).sort_values(by='Coefficient', ascending=True)
    gbmCoefs = gbmCoefs[gbmCoefs['Coefficient'] > 0.001]


    ut.plotDF(gbmCoefs, {'kind': 'barh', 'x': 'Variable','y': 'Coefficient', 'legend': False },
           {
            'xticks': xTickMultLS,
            'grid': None,
            'xlabel': 'Gradient Boost Coefficient',
            'ylabel': 'Variable Name',
            'title': 'Bar Plot of Gradient Boost Feature Selection ',
            'savefig': barDir + 'Gradient Boost Feature Selection.png'}, removeOutliersBeforePlotting=False)

    # ut.plotDF(train[['SalePrice']], {'kind': 'bar', 'x': ,'y': },
    #        {xTickFormatDollars: '',
    #         yTickFormat: '',
    #         'grid': None,
    #         'xlabel': 'Sale Price',
    #         'title': 'Histogram of Sale Price of Houses in Ames, Iowa',
    #         'savefig': histDir + 'SalePrice.png'})

    print('Finished Visualizations')

# plots:




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


