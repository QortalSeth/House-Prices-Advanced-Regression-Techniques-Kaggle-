from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
from Modules.Global import getColumnType, Regression
import pandas as pd
import numpy as np


# d. regression

class Regression:
    def __init__(self, model, type: str, hyperparams={}):
        self.model = model
        self.type = type
        self.hyperparams = hyperparams

    def fit(self,x,y):
        self.model.fit(x,y)


    def fitCV(self,x,y, cv=5):
        self.model.fit(x,y)
        self.grid = GridSearchCV(self.model, self.hyperparams, cv=cv, return_train_score = True, n_jobs=-1)
        fit = self.grid.fit(x,y)
        self.bestParams = fit.best_params_
        self.score = fit.best_score_

def performRegressions(modDF: pd.DataFrame):
    r = Regression(LinearRegression(), 'Linear')
    models = [

    Regression(LinearRegression(), 'Linear'),
    Regression(Ridge(), 'Ridge', {'alpha': np.linspace(1e-3,200,20)}),
    Regression(Lasso(), 'Lasso', {'alpha': np.linspace(1e-3,200,20)}),
    Regression(ElasticNet(), 'ElasticNet', {'Rho': np.linspace(0.01, 1, 30)}),

    Regression(RandomForestRegressor(), 'Random Forest',
    {'criterion': ['gini', 'entropy'],
    'max_depth': range(1, 31),
    'n_estimators': range(10, 110, 10)}),

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

    x = modDF.drop(columns=['SalePrice', 'LogSalePrice'])
    y = modDF['LogSalePrice']

    for model in models:
        model.fitCV(x,y)
    # nominal = modDF[getColumnType('Nominal')].copy()
    # ordinal = modDF[getColumnType('Ordinal')].copy()
    # discrete = modDF[getColumnType('Discrete')].copy()

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