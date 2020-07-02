from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LassoCV
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
from Modules.Global import *
import pandas as p
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import pandas as pd
import Modules.Util as ut
from Modules.Global import splitDfByCategory
from sklearn.preprocessing import StandardScaler
from Modules.EDA import train
from typing import Dict
from Modules.EDA import test
import matplotlib.pyplot as plt

import pickle


# d. regression

class Regression:
    def __init__(self, model, name: str, hyperparams={}):
        self.model = model
        self.name = name
        self.hyperparams = hyperparams
        self.bestParams: Dict
        self.time=''

    def fit(self, xTrain, xTest, yTrain, yTest):
        print('Starting ', self.name)

        self.model.fit(xTrain,yTrain)
        self.modelCV = self.model
        self.trainScore = self.model.score(xTrain, yTrain)
        self.testScore = self.model.score(xTest, yTest)
        self.trainRMSE = self.getRMSE(yTrain, self.model.predict(xTrain))
        self.testRMSE = self.getRMSE(yTest, self.model.predict(xTest))


    def fitCV(self,xTrain, xTest, yTrain, yTest, cv=5):
        print('Starting ', self.name)


        grid = GridSearchCV(self.model, self.hyperparams, cv=cv, return_train_score = False, n_jobs=-1)
        self.modelCV = grid.fit(xTrain,yTrain)
        self.bestParams = self.modelCV.best_params_
        self.trainScore = self.modelCV.best_estimator_.score(xTrain, yTrain)
        self.testScore = self.modelCV.best_estimator_.score(xTest, yTest)

        # if 'max_depth' in self.hyperparams:
        #     featureSelection = self.model.feature_importances_
        #     print(featureSelection)

        self.trainRMSE = self.getRMSE(yTrain, self.modelCV.predict(xTrain))
        self.testRMSE = self.getRMSE(yTest, self.modelCV.predict(xTest))


    def getRMSE(self, y, predicted):
        return sqrt(np.exp(mean_squared_error(y, predicted)))




    def plotHyperParams(self, trainX, testX, trainY, testY, i):


        for name, params in self.hyperparams.items():
            coefs = []
            intercepts = []
            trainScore = []
            testScore = []

            if len(params) < 2:
                continue

            for value in params:
                self.model.set_params(**{name: value})
                #print(name, '   Value: ',value)
                self.model.fit(trainX, trainY)
           # intercepts.append(self.model.intercept_)
           # coefs.append(self.model.coef_)
                trainScore.append(self.model.score(trainX, trainY))
                testScore.append(self.model.score(testX, testY))

            plt.plot(params, trainScore, label=r'train set $R^2$')
            plt.plot(params, testScore, label=r'test set $R^2$')

            plt.xlabel(name+' Value')
            plt.ylabel('R^2 Value')
            plt.title(self.name+' R^2 VS. '+ name)
            plt.legend(loc=4)
            plt.savefig('Output/Hyperparams/'+str(i)+' - '+self.name+' '+name+'.png')
            plt.clf()


def assembleModels():

    models = {
    'Linear'     : Regression(LinearRegression(n_jobs=-1), 'Linear'),
    'Ridge'      :  Regression(Ridge(), 'Ridge', {'alpha': np.linspace(1,6,100)}),
    'Lasso'      :  Regression(Lasso(), 'Lasso', {'alpha': np.linspace(1e-10,0.002,200)}),
    'Elastic Net': Regression(ElasticNet(), 'ElasticNet', {'alpha': np.linspace(1e-10, 10, 200), 'l1_ratio': np.linspace(0, 1, 10)}),

    'Random Forest': Regression(RandomForestRegressor(n_jobs=-1), 'Random Forest',
    {   'max_depth': range(5, 20),
        'n_estimators': range(20, 40, 2)}),

    'Gradient Boost': Regression(GradientBoostingRegressor(), 'Gradient Boost',
               {'learning_rate': np.linspace(.001, 0.1, 10),
                'n_estimators': range(60, 80, 5),
                'max_depth': range(1, 6),
                'loss': ['ls']}), # use feature_importances for feature selection

    'SVM': Regression(SVR(), 'Support Vector Regressor',
               {'C': np.linspace(1, 10, 30),
                'gamma': np.linspace(1e-7, 0.1, 30)})
    #Regression((), ''),
    #Regression((), ''),
    }
    return models

def performRegressions(df: pd.DataFrame):
    models = assembleModels()
    y = df['LogSalePrice']

    continuousColumns = getColumnType(df, 'Continuous', True)
    discreteColumns   = getColumnType(df, 'Discrete', True)
    ordinalColumns = getColumnType(df, 'Ordinal', True)


    continuousColumns.remove('LogSalePrice')
    x = scaleData(df.drop(columns=['LogSalePrice']), continuousColumns)
    #x = df.drop(columns=['LogSalePrice'])
    trainTestData = train_test_split(x, y, test_size=0.3, random_state=0)

    #models['Ridge'].plotHyperParams(*trainTestData, 1)
    # models['Lasso'].plotHyperParams(*trainTestData,2)
    # models['Elastic Net'].plotHyperParams(*trainTestData)

    # models['Ridge'].plotHyperParams(*trainTestData)
    # models['SVM'].plotHyperParams(*trainTestData)
    i=0
    for name, model in models.items():
        model.plotHyperParams(*trainTestData,i)
        i+=1

    models['Linear'].time,         returnValue = ut.getExecutionTime(lambda: models['Linear'].fit(*trainTestData))
    models['Ridge'].time,          returnValue = ut.getExecutionTime(lambda: models['Ridge'].fitCV(*trainTestData))
    models['Lasso'].time,          returnValue = ut.getExecutionTime(lambda: models['Lasso'].fitCV(*trainTestData))
    models['Elastic Net'].time,    returnValue = ut.getExecutionTime(lambda: models['Elastic Net'].fitCV(*trainTestData))

    models['Random Forest'].time,  returnValue = ut.getExecutionTime(lambda: models['Random Forest'].fitCV(*trainTestData))
    models['Gradient Boost'].time, returnValue = ut.getExecutionTime(lambda: models['Gradient Boost'].fitCV(*trainTestData))
    models['SVM'].time,            returnValue = ut.getExecutionTime(lambda: models['SVM'].fitCV(*trainTestData))

    results = pd.DataFrame([r.__dict__ for r in models.values()]).drop(columns=['model', 'modelCV'] )

    roundColumns4Digits = ['trainScore', 'testScore']
    #roundColumns8Digits = ['trainRMSE', 'testRMSE']
    for c in roundColumns4Digits:
        results[c] = results[c].apply(ut.roundTraditional, args = (4,) )

    results.to_excel('Output/Model Results.xlsx')
    print('Finished Regressions')
    return models





def predictSalePrice(dfTest: pd.DataFrame, models: Dict):
    continuousColumns = getColumnType(dfTest, 'Continuous', True)

    x = scaleData(dfTest, continuousColumns)
    predictions = pd.DataFrame()

    for regression in models.values():
        prediction = pd.Series(regression.model.predict(x))
        predictions = ut.appendColumns([predictions, prediction])

    salePrice  = predictions.apply(np.exp, axis=1)
    finalPrediction = salePrice.apply(np.mean, axis=1)

    output = pd.DataFrame({'Id': test['Id'].astype(int), 'SalePrice': finalPrediction})
    output.to_csv('Output/Submission.csv', index=False)














# look up randomizedSearchCV vs. GridsearchCV

# use VIF > 5, AIC, BIC for feature selection
# Don't use linear regression on categorical vars
# create ensemble of many different models (check for packages that can do this)
# use linear model on everything, then feature select