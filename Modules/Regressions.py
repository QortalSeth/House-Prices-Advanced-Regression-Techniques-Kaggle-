from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
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

import pickle


# d. regression

class Regression:
    def __init__(self, model, name: str, hyperparams={}):
        self.model = model
        self.name = name
        self.hyperparams = hyperparams
        self.bestParams: Dict
        self.time=''

    def fit(self,x,y):
        print('Starting ', self.name)
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=0)

        self.model.fit(xTrain,yTrain)
        self.trainScore = self.model.score(xTrain, yTrain)
        self.testScore = self.model.score(xTest, yTest)
        self.trainRMSE = self.getRMSE(yTrain, self.model.predict(xTrain))
        self.testRMSE = self.getRMSE(yTest, self.model.predict(xTest))



    def fitCV(self,x,y, cv=5):
        print('Starting ', self.name)
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=0)

        grid = GridSearchCV(self.model, self.hyperparams, cv=cv, return_train_score = True, n_jobs=-1)
        self.model = grid.fit(xTrain,yTrain)
        self.bestParams = self.model.best_params_
        self.trainScore = self.model.best_estimator_.score(xTrain, yTrain)
        self.testScore = self.model.best_estimator_.score(xTest, yTest)

        self.trainRMSE = self.getRMSE(yTrain, self.model.predict(xTrain))
        self.testRMSE = self.getRMSE(yTest, self.model.predict(xTest))


    def getRMSE(self, y, predicted):
        return sqrt(mean_squared_error(y, predicted))


def assembleModels():

    alpha = np.linspace(1e-10,50,50)
    models = {
    'Linear'     : Regression(LinearRegression(n_jobs=-1), 'Linear'),
    'Ridge'      :  Regression(Ridge(), 'Ridge', {'alpha': alpha}),
    'Lasso'      :  Regression(Lasso(), 'Lasso', {'alpha': alpha}),
    'Elastic Net': Regression(ElasticNet(), 'ElasticNet', {'alpha': alpha, 'l1_ratio': np.linspace(0, 1, 20)}),

    'Random Forest': Regression(RandomForestRegressor(n_jobs=-1), 'Random Forest',
    {   'max_depth': range(2, 20),
        'n_estimators': range(10, 60, 10)}),

    'Gradient Boost': Regression(GradientBoostingRegressor(), 'Gradient Boost',
               {'learning_rate': np.linspace(.001, 0.2, 10),
                'n_estimators': range(10, 100, 10),
                'max_depth': range(2, 10, 2),
                'loss': ['ls']}), # use feature_importances for feature selection

    'SVM': Regression(SVR(), 'Support Vector Regressor',
               {'C': np.linspace(1, 20, 20),
                'gamma': np.linspace(1e-6, 1e-2, 10)})
    #Regression((), ''),
    #Regression((), ''),
    }
    return models


def performRegressions(df: pd.DataFrame):
    models = assembleModels()
    y = df['LogSalePrice']

    continuousColumns = getColumnType(df, 'Continuous', True)

    continuousColumns.remove('LogSalePrice')
    x = scaleData(df.drop(columns=['LogSalePrice']), continuousColumns)

    models['Linear'].time,         returnValue = ut.getExecutionTime(lambda: models['Linear'].fit(x, y))
    models['Ridge'].time,          returnValue = ut.getExecutionTime(lambda: models['Ridge'].fitCV(x, y))
    models['Lasso'].time,          returnValue = ut.getExecutionTime(lambda: models['Lasso'].fitCV(x, y))
    models['Elastic Net'].time,    returnValue = ut.getExecutionTime(lambda: models['Elastic Net'].fitCV(x, y))

    models['Random Forest'].time,  returnValue = ut.getExecutionTime(lambda: models['Random Forest'].fitCV(x,y))
    models['Gradient Boost'].time, returnValue = ut.getExecutionTime(lambda: models['Gradient Boost'].fitCV(x, y))
    models['SVM'].time,            returnValue = ut.getExecutionTime(lambda: models['SVM'].fitCV(x, y))

    results = pd.DataFrame([r.__dict__ for r in models.values()]).drop(columns=['model', 'hyperparams'] )

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
    predictions = pd.DataFrame(test['Id'])

    for regression in models.values():

        # x is 1459 by 216
        prediction = regression.model.predict(x)
        predictions = ut.appendColumns([predictions, prediction])

    finalPrediction = predictions.apply(np.mean, axis=0).apply(np.exp)

    output = pd.concat([dfTest['Id'], finalPrediction])
    output.to_excel('../Output/Submission.xlsx')














# look up randomizedSearchCV vs. GridsearchCV

# use VIF > 5, AIC, BIC for feature selection
# Don't use linear regression on categorical vars
# create ensemble of many different models (check for packages that can do this)
# use linear model on everything, then feature select