from data_understanding import Data
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

"""Adding the Classification"""
#for 1400rpm
y = Data[Data["test_info"]==1400]["flap_pos_open"]
X = Data[Data["test_info"]==1400][["peak_bladepass","a_rms"]]
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=0.25, random_state=0)        
model = GaussianNB()
model.fit(Xtrain,ytrain)
ypred = model.predict(Xtrain)
correct = accuracy_score(ytrain,ypred, normalize=False)
incorrect = np.size(ytrain)-correct
print(f"1400rpm-peak_bladepass/a_rms: Score=W{incorrect}-R{correct} accuracy={accuracy_score(ytrain,ypred):.2f}%")
#for 2800rpm
y = Data[Data["test_info"]==2800]["flap_pos_open"]
X = Data[Data["test_info"]==2800][["peak_bladepass","a_rms"]]
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=0.25, random_state=0)        
model = GaussianNB()
model.fit(Xtrain,ytrain)
ypred = model.predict(Xtrain)
correct = accuracy_score(ytrain,ypred, normalize=False)
incorrect = np.size(ytrain)-correct
print(f"2800rpm-peak_bladepass/a_rms: Score=W{incorrect}-R{correct} accuracy={accuracy_score(ytrain,ypred):.2f}%")