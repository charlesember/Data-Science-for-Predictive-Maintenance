from os import times
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
import seaborn as sns

# Loading the Dataset
times = pd.read_csv("./Daten3-4/Drivers_reloaded.csv",header=2)
# Looking if all values are numerical 
print(times.describe())

# Replacing the non-numerical values with "NaN"
for columns in times.columns:
    times[columns]=pd.to_numeric(times[columns], errors="coerce")
    
# Checking for outliers
plt.figure()
sns.boxplot(data=times)
plt.show()

# Removing the outliers
for columns in times.columns:
    for i in range(len(times)):
        if times[columns][i] >= 100 or times[columns][i] <= 20:
            times[columns][i] = None

#Checking if everything was removed correctly          
plt.figure()
sns.boxplot(data=times)
plt.show()

#Dropping the NaN rows
times.dropna(inplace=True)

#Plotting the histogram
plt.figure()
sns.histplot(data=times, stat="probability")
plt.show()