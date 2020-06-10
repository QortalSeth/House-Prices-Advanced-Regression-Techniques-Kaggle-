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


# d. regression

class Regression:
    def __init__(self, model, name: str, hyperparams={}):
        self.model = model
        self.name = name
        self.hyperparams = hyperparams

    def fit(self,x,y):
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=0)

        self.model.fit(xTrain,yTrain)
        self.score = self.model.score(xTest, yTest)
        self.predicted = self.model.predict(xTest)
        self.getRMSE(yTest)



    def fitCV(self,x,y, cv=5):
        self.model.fit(x,y)
        self.grid = GridSearchCV(self.model, self.hyperparams, cv=cv, return_train_score = True, n_jobs=-1)
        fit = self.grid.fit(x,y)
        self.bestParams = fit.best_params_
        self.score = fit.best_score_
        self.predicted = self.model.predict(x)
        self.getRMSE(y)

    def getRMSE(self, y):
        self.rmse = sqrt(mean_squared_error(y, self.predicted))


def assembleModels():
    mi = 1000

    models = [
    Regression(LinearRegression(n_jobs=-1), 'Linear'),
    Regression(Ridge(), 'Ridge', {'alpha': np.linspace(1e-3,200,20)}),
    Regression(Lasso(max_iter = mi), 'Lasso', {'alpha': np.linspace(1e-3,200,20)}),
    Regression(ElasticNet(max_iter = mi), 'ElasticNet', {'l1_ratio': np.linspace(0.01, 1, 30)}),

    Regression(RandomForestRegressor(n_jobs=-1), 'Random Forest',
    {
        'max_depth': range(1, 21),
        'n_estimators': range(10, 60, 10)}),

    Regression(GradientBoostingRegressor(), 'Gradient Boost',
               {'learning_rate': np.linspace(.001, 1, 20),
                'n_estimators': np.linspace(10, 1000, 50),
                'max_depth': range(2, 12, 2),
                'loss': ['ls', 'lad', 'huber', 'quantile']}
    ), # use feature_importances for feature selection

    Regression(SVR(), 'Support Vector Regressor',
               {'C': np.linspace(1, 100, 20),
                'gamma': np.linspace(1e-4, 1e-2, 10)})
    #Regression((), ''),
    #Regression((), ''),
    ]
    return models


def performRegressions(df: pd.DataFrame):
    models = assembleModels()
    continuousColumns = getColumnType('Continuous', True)
    continuousColumns = [x for x in continuousColumns if x in df]
    scaledDF = scaleData(df, continuousColumns)


    x = scaledDF.drop(columns=['LogSalePrice'])
    y = df['LogSalePrice']

    for model in models:
        paramDict = {'x': x, 'y': y}
        if model.hyperparams:
            print('Starting ', model.name)
            ut.getExecutionTime(model.fitCV, paramDict)
            print('')
        else:
            print('Starting ', model.name)
            ut.getExecutionTime(model.fit, paramDict)
            print('')

    df = pd.DataFrame([r.__dict__ for r in models])
    return models, df




# look up randomizedSearchCV vs. GridsearchCV
# perform linear regression on all features vs sales price
# use box cox transformation on numeric vars when doing linear models
# use spline regression as a ML model
# use XGBoost, GBM, Random Forest, tree models
#
# use VIF > 5, AIC, BIC for feature selection
# Don't use linear regression on categorical vars
# create ensemble of many different models (check for packages that can do this)
# use linear model on everything, then feature select