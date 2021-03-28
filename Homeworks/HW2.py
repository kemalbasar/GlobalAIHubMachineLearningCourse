import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.datasets import load_boston
import seaborn as sns
import pprint


Xb, yb = load_boston(return_X_y=True)
pd_boston = pd.DataFrame(Xb, columns=load_boston().feature_names)
pd_boston["price"] = yb
type(yb)



pd_boston.info()
statistics_of_boston_data = pd_boston.describe()


# This for displaying describe dataframe at the pycharm scientific data representation.
statistics_of_boston_data.index = statistics_of_boston_data.index.str.replace('%', 'p')

pd_boston.duplicated()
pd_boston.isna().sum()

sns.pairplot(pd_boston)
# This for displaying pilots at scientific view.
plt.show()

sns.distplot(pd_boston.AGE)
plt.show()

sns.distplot(pd_boston.CRIM)
plt.show()

sns.distplot(pd_boston.INDUS)
plt.show()

# Pycharm already shows heat map of data at scientific view.
# Correlated features should be omited before removing isolated data.
# Detecting correlations and droping appropraite features from data from

correlation_frame = pd_boston.corr()
features = pd_boston.columns
correlated_columns = dict.fromkeys(features)
correlated_columns_list = []

for item2 in correlation_frame.index:
    for item in correlation_frame.index:
        if correlation_frame[item2][item] < -0.75 or correlation_frame[item2][item] > 0.75:
            if correlation_frame[item2][item] != -1 and correlation_frame[item2][item] != 1:
             correlated_columns_list.append(item)
    correlated_columns[item2] = correlated_columns_list
    correlated_columns_list = []

pd_boston.drop(labels=["INDUS", "DIS", "RAD"], axis=1, inplace=True)


model = IsolationForest(contamination=0.08)
isolate_index = model.fit_predict(pd_boston)
pd_boston["isolate index"] = isolate_index
index_names = pd_boston[(pd_boston["isolate index"] == -1)].index

pd_boston.drop(labels=index_names, axis=0, inplace=True)
pd_boston.reset_index(inplace=True)
pd_boston.drop(labels="index", axis=1, inplace=True)
pd_boston.drop(labels="isolate index", axis=1, inplace=True)

Xb = pd_boston.iloc[:, 0:-1].to_numpy()
Yb = pd_boston.iloc[:, -1:].to_numpy()
len(Xb)
len(Yb)


X_train, X_test, y_train, y_test = train_test_split(Xb, Yb, test_size=0.3, random_state=42)

def adj_r2 (X,y,model):
    """
    X: input
    y: output
    model: regression model
    """
    r_squared = model.score(X,y)
    return 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - X.shape[1] - 1)

regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

def model_to_score_ridge(alpha,setx,sety,testx,testy):
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(setx, sety)
    scoreridge = [adj_r2(setx,sety,ridge_model),adj_r2(testx,testy,ridge_model)]
    return scoreridge
def model_to_score_lasso(alpha,setx,sety,testx,testy):
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(setx, sety)
    scorelasso = [adj_r2(setx,sety,lasso_model),adj_r2(testx,testy,lasso_model)]
    return scorelasso


# Simple Linear Model
print("Simple Train: ", adj_r2(X_train,y_train,regression_model))
print("Simple Test: ", adj_r2(X_test,y_test,regression_model))
print('*************************')
# Lasso
print("Lasso Train: ",model_to_score_lasso(0.01,X_train,y_train,X_test,y_test))
print("Lasso Train: ",model_to_score_lasso(0.05,X_train,y_train,X_test,y_test))
print("Lasso Train: ",model_to_score_lasso(0.2,X_train,y_train,X_test,y_test))
print("Lasso Train: ",model_to_score_lasso(0.5,X_train,y_train,X_test,y_test))
print("Lasso Train: ",model_to_score_lasso(1,X_train,y_train,X_test,y_test))

print('*************************')
# Ridge
print("Ridge Train: ", model_to_score_ridge(1,X_train,y_train,X_test,y_test))
print("Ridge Train: ",model_to_score_ridge(10,X_train,y_train,X_test,y_test))
print("Ridge Train: ",model_to_score_ridge(20,X_train,y_train,X_test,y_test))
print("Ridge Train: ",model_to_score_ridge(50,X_train,y_train,X_test,y_test))
print("Ridge Train: ",model_to_score_ridge(500,X_train,y_train,X_test,y_test))


#Comment : lower alpha values make better prediction for our data set in both regularization models.Difference between regularization models and simple lineer regression model
# reducing but never comes to zero. Linear regression model is better for that dataset.
