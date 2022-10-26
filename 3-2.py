import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris") 

print(iris.head(10))
print()

norm = MinMaxScaler().fit_transform(iris.drop(columns="species"))
stand = StandardScaler().fit_transform(iris.drop(columns="species"))

iris_norm = pd.DataFrame(norm, columns= ["sepal_length","sepal_width","petal_length","petal_width"])
iris_stand = pd.DataFrame(stand, columns= ["sepal_length","sepal_width","petal_length","petal_width"])

pd.set_option("float_format", "{:.3f}".format)
print(iris.describe()) 
print(iris_norm.describe())
print(iris_stand.describe())

iris.plot(style="*")
iris_norm.plot(style="*")
iris_stand.plot(style="*")

plt.figure()
sns.boxplot(data=iris)
plt.figure()
sns.boxplot(data=iris[iris.species == "versicolor"])

plt.figure()
sns.histplot(data=iris[iris.species == "versicolor"], x="petal_length")



import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris") 

print(iris.head(10))
print()

norm = MinMaxScaler().fit_transform(iris.drop(columns="species"))
stand = StandardScaler().fit_transform(iris.drop(columns="species"))

iris_norm = pd.DataFrame(norm, columns= ["sepal_length","sepal_width","petal_length","petal_width"])
iris_stand = pd.DataFrame(stand, columns= ["sepal_length","sepal_width","petal_length","petal_width"])

pd.set_option("float_format", "{:.3f}".format)
print(iris.describe()) 
print(iris_norm.describe())
print(iris_stand.describe())

iris.plot(style="*")
iris_norm.plot(style="*")
iris_stand.plot(style="*")

plt.figure()
sns.boxplot(data=iris)
plt.figure()
sns.boxplot(data=iris[iris.species == "versicolor"])

plt.figure()
sns.histplot(data=iris[iris.species == "versicolor"], x="petal_length")
