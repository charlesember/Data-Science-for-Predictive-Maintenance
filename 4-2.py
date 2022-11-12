from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

data = pd.read_csv("./Daten4-2/jump_distance.csv", header=0)

""""Linear regression part of the exercise"""
# model = LinearRegression(fit_intercept=True)
# x = data["Mean wind speed in m/s"][data["Weight of biker in kg"] == 70].to_numpy()
# X = x[:,np.newaxis]
# y = data["Jump distance in m"][data["Weight of biker in kg"] == 70].to_numpy()
# model.fit(X,y)
# ypred = model.predict(X)

# print(f"R2_score = {r2_score(y,ypred):.3f}")
# print(f"Coeff = {model.coef_[0]:.3f}")
# print(f"Intercept = {model.intercept_:.3f}")

# plt.figure()
# plt.scatter(x,y)
# plt.plot(x, ypred)
# plt.show()

"""Quadratic regression part of the exercise"""
# polymod = PolynomialFeatures(degree=2, include_bias=False)
# model = LinearRegression(fit_intercept=True)
# x = data["Weight of biker in kg"][data["Mean wind speed in m/s"] == 10].to_numpy()
# y = data["Jump distance in m"][data["Mean wind speed in m/s"] == 10].to_numpy()
# print(y)
# X = x[:,np.newaxis]
# X2 = polymod.fit_transform(X)
# print(X2)
# model.fit(X2,y)
# y2pred = model.predict(X2)

# print(f"R2_score = {r2_score(y,y2pred):.3f}")
# print(f"Coeff = {model.coef_[0]:.3f}, {model.coef_[1]:.3f}")
# print(f"Intercept = {model.intercept_:.3f}")

# plt.figure()
# plt.scatter(X2[:,0],y)
# plt.plot(X2[:,0], y2pred)
# plt.show()

"""Multivariate regression part of the exercise"""
x1 = data["Weight of biker in kg"].to_numpy()
x2 = data["Mean wind speed in m/s"].to_numpy()
y = data["Jump distance in m"].to_numpy()
X = np.stack([x1,x2], axis=1)

model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression(fit_intercept=True))
model.fit(X,y,)
ypred = model.predict(X)

print(f"R2_score = {r2_score(y,ypred):.3f}")
plt.figure()
plt.scatter(x1, x2, c=y)
cbar = plt.colorbar()
cbar.set_label("Results")
plt.show()

preddata = np.array([[63,14.7]])
y2pred = model.predict(preddata)
print(y2pred)
