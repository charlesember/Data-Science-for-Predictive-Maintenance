from os import times
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
import seaborn as sns

data = pd.read_csv("./Daten3-1/Drivers_new.csv", sep=",", header=0)

cosy_norm = MinMaxScaler().fit_transform(data[["cosy"]])
sporty_norm = MinMaxScaler().fit_transform(data[["sporty"]])
cosy_stand = StandardScaler().fit_transform(data[["cosy"]])
sporty_stand = StandardScaler().fit_transform(data[["sporty"]])

data.insert(2,"cosy_norm", cosy_norm)
data.insert(3,"sporty_norm", sporty_norm)
data.insert(4,"cosy_stand", cosy_stand)
data.insert(5,"sporty_stand", sporty_stand)

pd.set_option("float_format", "{:.3f}".format)
print(data.describe())

fig, axes = plt.subplots(nrows=2, ncols=2)

data[["cosy","sporty"]].plot(style="*", ax=axes[0,0])
data[["cosy_norm","sporty_norm"]].plot(style="*", ax=axes[1,0])
data[["cosy_stand","sporty_stand"]].plot(style="*", ax=axes[1,1])
data[["cosy_norm","sporty_norm","cosy_stand","sporty_stand"]].boxplot(ax=axes[0,1])
plt.show()
