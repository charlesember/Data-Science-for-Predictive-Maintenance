from os import times
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
import seaborn as sns

# Loading the Dataset
times = pd.read_csv("./Daten3-3/Drivers_again.csv",header=1)
# Looking if all values are numerical 
print(times.describe())

# Replacing the non-numerical values with "NaN"
for columns in times.columns:
    times[columns]=pd.to_numeric(times[columns], errors="coerce")

# printing the new values
print(times.describe())

# Checking for outliers
plt.figure()
sns.boxplot(data=times)
plt.show()

# Removing the outliers
for columns in times.columns:
    for i in range(len(times)):
        if times[columns][i] >= 100 or times[columns][i] <= 30:
            times[columns][i] = None

# Implementing Normal and Standart distributions
cosy_norm = MinMaxScaler().fit_transform(times[["cosy"]])
sporty_norm = MinMaxScaler().fit_transform(times[["sporty"]])
cosy_stand = StandardScaler().fit_transform(times[["cosy"]])
sporty_stand = StandardScaler().fit_transform(times[["sporty"]])

# Inserting Normal and Standart distributions
times.insert(2,"cosy_norm", cosy_norm)
times.insert(3,"sporty_norm", sporty_norm)
times.insert(4,"cosy_stand", cosy_stand)
times.insert(5,"sporty_stand", sporty_stand)

# Printing the new values 
print(times.describe())
plt.plot()
sns.boxplot(data=times)
plt.show()

# Dropping the NaN Values
times.dropna(inplace=True)

# Convert into NumPy
times_NP = times.to_numpy()

#Calculate the mean values
i=0
for cols in times.columns:
    globals()[f"{cols}_mean"] = np.mean(times_NP[:,i])
    print(f"{cols}_mean" + " = " + str(globals()[f"{cols}_mean"]))
    i=i+1
print(times.describe())
