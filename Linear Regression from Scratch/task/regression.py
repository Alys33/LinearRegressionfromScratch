# required modules
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#my custom linear Regression
class CustomLinearRegression:
    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = ...
        self.intercept = ...

    def fit(self, X, y):
        X = X.to_numpy()

        if self.fit_intercept:
            b = np.array([1 for _ in range(len(X))]).reshape(len(X), 1)
            X = np.hstack((b, X))
            coef = np.linalg.inv(X.T@X)@(X.T@y)
            self.coefficient = coef[1:]
            self.intercept = coef[0]
        else:
            self.coefficient = np.linalg.inv(X.T@X)@(X.T@y)

    def predict(self, X):
        X = X.to_numpy()
        if self.fit_intercept:
            return X@self.coefficient.T + self.intercept
        else:
            return X@self.coefficient.T

    def r2_score(self, y, yhat):
        y = y.to_numpy()
        y_mean = y.mean()
        return 1 - sum((y - yhat) ** 2) / sum((y - y_mean) ** 2)

    def rmse(self, y, yhat):
        y = y.to_numpy()
        mse = sum((y - yhat) ** 2) / len(y)
        return np.sqrt(mse)



# Stage  1/4 : Fit method

file = {"x": [4.0, 4.5, 5, 5.5, 6.0, 6.5, 7.0],
        "y": [33, 42, 45, 51, 53, 61, 62]}

df = pd.DataFrame(file)
#my custom linear Regression





# Fitting to my custom linear regression
# myLinear = CustomLinearRegression(fit_intercept=True)
# myLinear.fit(df["x"], df["y"])
#
# dict = {"Intercept": myLinear.intercept, "Coefficient": myLinear.coefficient}
# print(dict)



# Stage 2/4 : Multiple linear regression and predictions


# mydict = {"X": [4, 4.5, 5, 5.5, 6, 6.5, 7],
#           "W": [1, -3, 2, 5, 0, 3, 6],
#           "Z": [11, 15, 12, 9, 18, 13, 16],
#           "y": [33, 42, 45, 51, 53, 61, 62]}
# df2 = pd.DataFrame(mydict)
#
# reg = CustomLinearRegression(fit_intercept=True)
# reg.fit(df2[['X', "W", "Z"]], df2['y'])
# y_pred = reg.predict(df2[['X', "W", "Z"]])
# print(y_pred)

# Stage 3/4: Metrics implementation
#
# dict = {"Capacity": [0.9, 0.5, 1.75, 2.0,1.4, 1.5, 3.0, 1.1, 2.6,1.9],
#         "Age": [11, 11, 9, 8, 7, 7, 6, 5, 5, 4],
#         "Cost/ton": [21.95, 27.18, 16.9, 15.37, 16.03, 18.15, 14.22, 18.72, 15.4, 14.69]}
#
# data = pd.DataFrame(dict)
#
# my_reg = CustomLinearRegression(fit_intercept=True)
# X = data[['Capacity', 'Age']]
# y = data['Cost/ton']
# my_reg.fit(X, y)
#
# y_pred = my_reg.predict(X)
# Rmse = my_reg.rmse(y, y_pred)
# r_score = my_reg.r2_score(y, y_pred)
# dict_values = {"Intercept": my_reg.intercept,
#                "Coefficient": my_reg.coefficient,
#                "R2": r_score,
#                "RMSE": Rmse}
#
# print(dict_values)



# Stage 4/4: Scikit-Learn linear regression
data = pd.read_csv("data_stage4.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

regSci = LinearRegression(fit_intercept=True)
regCus = CustomLinearRegression(fit_intercept=True)
regSci.fit(X, y)
regCus.fit(X, y)

y_pred_S = regSci.predict(X)
y_pred_C = regCus.predict(X)

rmse_S = np.sqrt(mean_squared_error(y,y_pred_S))
r_scor_S = r2_score(y, y_pred_S)


rmse_C = regCus.rmse(y, y_pred_C)
r_scor_C = regCus.r2_score(y,y_pred_C)

dict = {"Intercept": regSci.intercept_ - regCus.intercept,
        "Coefficient": regSci.coef_ - regCus.coefficient,
        "R2": r_scor_S - r_scor_C,
        "RMSE": rmse_S - rmse_C}
print(dict)

