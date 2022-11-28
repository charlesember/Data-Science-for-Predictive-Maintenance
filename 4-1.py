from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import NuSVC
import sklearn as skl
from sklearn.metrics import accuracy_score
import seaborn as sns
import pandas as pd
import numpy as np

# Part 1 of the Exercise
iris = sns.load_dataset("iris")
iris = pd.concat([iris[iris["species"] == "versicolor"], iris[iris["species"] == "virginica"]])
y = iris["species"]
print(iris.columns)
for col1 in iris.columns:
    for col2 in iris.columns:
        if col1 != col2 and col1!= "species" and col2 != "species":
            X = iris[[col1,col2]]

            Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=0)

            model = GaussianNB()
            model.fit(Xtrain, ytrain)
            ypred = model.predict(Xtrain)
            correct = accuracy_score(ytrain, ypred, normalize=False)
            incorrect = np.size(ytrain) - correct
            # print(f"{col1}/{col2}    Score: W-{incorrect}/R-{correct}    accuracy = {accuracy_score(ytrain, ypred):.2f}%")
# The best way to differenciate seems to be petal_width/petal_length

# Part 2 of the Exercise
X = iris[["petal_width", "petal_length"]]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=0)
models = []
models.append(KNeighborsClassifier())
models.append(LinearDiscriminantAnalysis())
models.append(DecisionTreeClassifier())
models.append(NuSVC())

for model in models:
    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtrain)
    correct = accuracy_score(ytrain, ypred, normalize=False)
    incorrect = np.size(ytrain) - correct
    print(f"Score: W-{incorrect}/R-{correct}    accuracy = {accuracy_score(ytrain, ypred):.2f}%")