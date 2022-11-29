from data_understanding import Data
from data_understanding import dataheader
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.metrics import r2_score

flowrates1400 ={
    0:0,
    25:4.15,
    50:8.2,
    75:12.38,
    100:16.46    
}
flowrates2800 = {
    0:0,
    25:7.96,
    50:15.97,
    75:24.24,
    100:32.38    
}
flowlis=[]

#Adding the flowrate Data into the Dataframe
for keys in flowrates1400.keys():
    for i in range(len(Data[(Data["test_info"]==1400) & (Data["flap_pos_open"]==keys)])):
        flowlis.append(flowrates1400[keys])
    for i in range(len(Data[(Data["test_info"]==2800) & (Data["flap_pos_open"]==keys)])):
        flowlis.append(flowrates2800[keys])
Data["Volume_Flow"] = flowlis

#Regression with linear model
for turnrates in Data["test_info"].unique():
    for header in dataheader:
        model = LinearRegression(fit_intercept=True)
        x = Data[header][Data["test_info"]==turnrates].to_numpy()
        X = x[:,np.newaxis]
        y = Data["Volume_Flow"][Data["test_info"]==turnrates].to_numpy()
        model.fit(X,y)
        ypred = model.predict(X)

        print(f"{turnrates} - {header}")
        print(f"r2_score = {r2_score(y,ypred):.3f}")
        print(f"Coeff = {model.coef_[0]:.3f}")
        print(f"intercept = {model.intercept_:.3f}")
        print()

#Regression with quadratic model
for turnrates in Data["test_info"].unique():
    for header in dataheader:
        polymod = PolynomialFeatures(degree=2, include_bias=False)
        model = LinearRegression(fit_intercept=True)
        x = Data[header][Data["test_info"]==turnrates].to_numpy()
        X = x[:,np.newaxis]
        X2 = polymod.fit_transform(X)
        y = Data["Volume_Flow"][Data["test_info"]==turnrates].to_numpy()
        model.fit(X2,y)
        ypred = model.predict(X2)

        print(f"{turnrates} - {header}")
        print(f"r2_score = {r2_score(y,ypred):.3f}")
        print(f"Coeff = {model.coef_[1]:.3f},{model.coef_[0]:.3f}")
        print(f"intercept = {model.intercept_:.3f}")
        print()