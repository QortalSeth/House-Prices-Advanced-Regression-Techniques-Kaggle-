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

# d. regression

class Regression:
    def __init__(self, model, name: str, hyperparams={}):
        self.model = model
        self.name = name
        self.hyperparams = hyperparams
        self.bestParams: Dict

    def fit(self,x,y):
        print('Starting ', self.name)
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=0)

        self.model.fit(xTrain,yTrain)
        self.score = self.model.score(xTest, yTest)
        self.predicted = self.model.predict(xTest)
        self.getRMSE(yTest)


    def fitCV(self,x,y, cv=5):
        print('Starting ', self.name)
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=0)

        self.model.fit(xTrain,yTrain)
        grid = GridSearchCV(self.model, self.hyperparams, cv=cv, return_train_score = True, n_jobs=-1)
        fit = grid.fit(xTest,yTest)
        self.bestParams = fit.best_params_
        self.score = fit.best_score_
        self.predicted = self.model.predict(x)
        self.getRMSE(y)

    def getRMSE(self, y):
        self.rmse = sqrt(mean_squared_error(y, self.predicted))


def assembleModels():

    alpha = np.linspace(1e-4,200,20)
    models = {
    'Linear'     : Regression(LinearRegression(n_jobs=-1), 'Linear'),
    'Ridge'      :  Regression(Ridge(), 'Ridge', {'alpha': alpha}),
    'Lasso'      :  Regression(Lasso(max_iter = 100000), 'Lasso', {'alpha': alpha}),
    'Elastic Net': Regression(ElasticNet(max_iter = 100000), 'ElasticNet', {'alpha': alpha, 'l1_ratio': np.linspace(0.01, 1, 20)}),

    'Random Forest': Regression(RandomForestRegressor(n_jobs=-1), 'Random Forest',
    {   'max_depth': range(2, 16),
        'n_estimators': range(10, 60, 10)}),

    'Gradient Boost': Regression(GradientBoostingRegressor(), 'Gradient Boost',
               {'learning_rate': np.linspace(.001, 1, 10),
                'n_estimators': range(10, 100, 10),
                'max_depth': range(2, 12, 2),
                'loss': ['ls']}), # use feature_importances for feature selection

    'SVM': Regression(SVR(), 'Support Vector Regressor',
               {'C': np.linspace(1, 200, 20),
                'gamma': np.linspace(1e-4, 1e-2, 10)})
    #Regression((), ''),
    #Regression((), ''),
    }
    return models


def performRegressions(df: pd.DataFrame, dummies: pd.DataFrame):
    models = assembleModels()
    continuousColumns = getColumnType(df, 'Continuous', True)
    scaledDF = scaleData(df, continuousColumns)

    dummyX = pd.concat([dummies, df], axis=1)
    nominal = train[ getColumnType(train, 'Nominal')]

    x = pd.concat([nominal , scaledDF.drop(columns=['LogSalePrice'])], axis=1)
    y = df['LogSalePrice']

    ut.getExecutionTime(lambda: models['Linear'].fit(dummyX, y))
    ut.getExecutionTime(lambda: models['Ridge'].fitCV(dummyX, y))
    ut.getExecutionTime(lambda: models['Lasso'].fitCV(dummyX, y))
    ut.getExecutionTime(lambda: models['Elastic Net'].fitCV(dummyX, y))

    ut.getExecutionTime(lambda: models['Random Forest'].fitCV(dummyX,y))
    ut.getExecutionTime(lambda: models['Gradient Boost'].fitCV(dummyX, y))
    ut.getExecutionTime(lambda: models['SVM'].fitCV(dummyX, y))

    results = pd.DataFrame([r.__dict__ for r in models.values()]).drop(columns=['model', 'hyperparams', 'predicted'] )
    return models, results




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