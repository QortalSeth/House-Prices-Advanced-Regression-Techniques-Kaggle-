from Modules.dfMods import modDF
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV


# d. regression
class Regression:

    def __init___(self, model, hyperparams={}):
        self.model = model
        self.hyperparams = hyperparams

    def fit(self,x,y):
        self.model.fit(x,y)


    def fitCV(self,x,y, cv=5):
        self.model.fit(x,y)
        self.grid = GridSearchCV(self.model, self.hyperparams, cv=cv, return_train_score = rts)



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