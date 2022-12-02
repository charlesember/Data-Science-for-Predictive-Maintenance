from data_understanding import Data
from data_understanding import dataheader
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline

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

"""Trying Multivariate Regression"""
#Regression with linear model
# print("----Linear Model Multivar----\n")
# for turnrates in Data["test_info"].unique():
#     for header in dataheader:
#         for header2 in dataheader:
#             if header!=header2:
#                 model = LinearRegression(fit_intercept=True)
#                 x = Data[header][Data["test_info"]==turnrates].to_numpy()
#                 x2 = Data[header2][Data["test_info"]==turnrates].to_numpy()
#                 X = np.stack([x, x2], axis=1)
#                 y = Data["Volume_Flow"][Data["test_info"]==turnrates].to_numpy()
#                 model.fit(X,y)
#                 ypred = model.predict(X)

#                 print(f"{turnrates} - {header}/{header2}")
#                 print(f"r2_score = {r2_score(y,ypred):.3f}")
#                 print()
        
#Regression with quadratic model
# print("\n----Quadratic Model----\n")
# for turnrates in Data["test_info"].unique():
#     for header in dataheader:
#         for header2 in dataheader:
#             if header != header2:
#                 polymod = PolynomialFeatures(degree=2, include_bias=False)
#                 model = make_pipeline(polymod, LinearRegression(fit_intercept=True))
#                 x = Data[header][Data["test_info"]==turnrates].to_numpy()
#                 x2 = Data[header2][Data["test_info"]==turnrates].to_numpy()
#                 X = np.stack([x, x2], axis=1)
#                 y = Data["Volume_Flow"][Data["test_info"]==turnrates].to_numpy()
#                 model.fit(X,y)
#                 ypred = model.predict(X)

#                 print(f"{turnrates} - {header}/{header2}")
#                 print(f"r2_score = {r2_score(y,ypred):.3f}")
#                 print()
#The biggest reachable Values are:
#0.937 for 1400 (peak_bladepass/a_rms)
#0.837 for 2800 (peak_bladepass/a_rms)

print("\n----Quadratic Model----\n")
for turnrates in Data["test_info"].unique():
    polymod = PolynomialFeatures(degree=2, include_bias=False)
    model = make_pipeline(polymod, LinearRegression(fit_intercept=True))
    x = Data["peak_bladepass"][Data["test_info"]==turnrates].to_numpy()
    x2 = Data["a_rms"][Data["test_info"]==turnrates].to_numpy()
    X = np.stack([x, x2], axis=1)
    y = Data["Volume_Flow"][Data["test_info"]==turnrates].to_numpy()
    model.fit(X,y)
    ypred = model.predict(X)

    print(f"{turnrates} - peak_bladepass/a_rms")
    print(f"r2_score = {r2_score(y,ypred):.3f}")
    print()



"""Neither linear nor quadratic regression provides usable Data"""
#Regression with linear model
# print("----Linear Model----\n")
# for turnrates in Data["test_info"].unique():
#     for header in dataheader:
#         model = LinearRegression(fit_intercept=True)
#         x = Data[header][Data["test_info"]==turnrates].to_numpy()
#         X = x[:,np.newaxis]
#         y = Data["Volume_Flow"][Data["test_info"]==turnrates].to_numpy()
#         model.fit(X,y)
#         ypred = model.predict(X)

#         print(f"{turnrates} - {header}")
#         print(f"r2_score = {r2_score(y,ypred):.3f}")
#         print(f"Coeff = {model.coef_[0]:.3f}")
#         print(f"intercept = {model.intercept_:.3f}")
#         print()
#Regression with quadratic model
# print("\n----Quadratic Model----\n")
# for turnrates in Data["test_info"].unique():
#     for header in dataheader:
#         polymod = PolynomialFeatures(degree=2, include_bias=False)
#         model = LinearRegression(fit_intercept=True)
#         x = Data[header][Data["test_info"]==turnrates].to_numpy()
#         X = x[:,np.newaxis]
#         X2 = polymod.fit_transform(X)
#         y = Data["Volume_Flow"][Data["test_info"]==turnrates].to_numpy()
#         model.fit(X2,y)
#         ypred = model.predict(X2)

#         print(f"{turnrates} - {header}")
#         print(f"r2_score = {r2_score(y,ypred):.3f}")
#         print(f"Coeff = {model.coef_[1]:.3f},{model.coef_[0]:.3f}")
#         print(f"intercept = {model.intercept_:.3f}")
#         print()