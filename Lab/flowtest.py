import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from scipy.fft import fft
import numpy as np
import calcs as cal
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

n1400rpm = {
    "0%": [],
    "25%": [],
    "50%": [],
    "75%": [],
    "100%": []
}
n2800rpm = {
    "0%": [],
    "25%": [],
    "50%": [],
    "75%": [],
    "100%": []
}
FFTn1400rpm = {
    "0%": [],
    "25%": [],
    "50%": [],
    "75%": [],
    "100%": []
}
FFTn2800rpm = {
    "0%": [],
    "25%": [],
    "50%": [],
    "75%": [],
    "100%": []
}
RMSVal={}
FFTn1400rpmc = {}
FFTn2800rpmc = {}  
"""Loading the data and making it ready to be concatenated"""
#loading the Data
timecols = ["Time", "Amplitude","B","C"]
for key in n1400rpm.keys():
    for filepath in listdir("./Data/Flowtest/"+key+"Flow/1400rpm"):
        n1400rpm[key].append(pd.read_csv("./Data/Flowtest/"+key+"Flow/1400rpm/" + filepath, header=None, sep=";", skiprows=1, names=timecols))
for key in n2800rpm.keys():
    for filepath in listdir("./Data/Flowtest/"+key+"Flow/2800rpm"):
        n2800rpm[key].append(pd.read_csv("./Data/Flowtest/"+key+"Flow/2800rpm/" + filepath, header=None, sep=";", skiprows=1, names=timecols))

#removing all NaN values and some unneccesary Columns 
for key in n1400rpm.keys():
    for i in range(len(n1400rpm[key])):
        n1400rpm[key][i].drop(columns=["B","C"], inplace=True)
        n1400rpm[key][i].dropna(inplace=True)
    for i in range(len(n2800rpm[key])):
        n2800rpm[key][i].drop(columns=["B","C"], inplace=True)
        n2800rpm[key][i].dropna(inplace=True)

"""Convert the dataframe into an numpy array"""
for keys in n1400rpm.keys():
    for i in range(len(n1400rpm[keys])):
        n1400rpm[keys][i]=n1400rpm[keys][i].to_numpy()
    for i in range(len(n2800rpm[keys])):
        n2800rpm[keys][i]=n2800rpm[keys][i].to_numpy()

"""Making the FFT spectre"""
fs = 6400
N = np.size(n1400rpm["0%"][1][:,1])
X = fs/N*np.arange(0,N//2)
for keys in FFTn1400rpm:
    for i in range(len(n1400rpm[keys])):
        FFTn1400rpm[keys].append(fft(n1400rpm[keys][i][:,1]))
        FFTn1400rpm[keys][i] = np.abs(FFTn1400rpm[keys][i][0:N//2])/N
        FFTn1400rpm[keys][i][1:]=2*FFTn1400rpm[keys][i][1:]
    for  i in range(len(n2800rpm[keys])):
        FFTn2800rpm[keys].append(fft(n2800rpm[keys][i][:,1]))
        FFTn2800rpm[keys][i] = np.abs(FFTn2800rpm[keys][i][0:N//2])/N
        FFTn2800rpm[keys][i][1:]=2*FFTn2800rpm[keys][i][1:]

#Remaking the Dataframes
for keys in n1400rpm.keys():
    for i in range(len(n1400rpm[keys])):
        n1400rpm[keys][i] = pd.DataFrame(n1400rpm[keys][i], columns=["Time","Amplitude"])
        FFTn1400rpm[keys][i] = pd.DataFrame(FFTn1400rpm[keys][i], columns=["Amplitude"])
    for i in range(len(n2800rpm[keys])):
        n2800rpm[keys][i] = pd.DataFrame(n2800rpm[keys][i], columns=["Time","Amplitude"])
        FFTn2800rpm[keys][i] = pd.DataFrame(FFTn2800rpm[keys][i], columns=["Amplitude"])

"""Adding up the FFT spectres"""
for key in FFTn1400rpm.keys():
    #Giving the index a name, so that it can be used to concatenate
    for i in range(len(FFTn1400rpm[key])):
        FFTn1400rpm[key][i]["Index"] = range(len(FFTn1400rpm[key][i]))
    for i in range(len(FFTn2800rpm[key])):
        FFTn2800rpm[key][i]["Index"] = range(len(FFTn2800rpm[key][i]))
    FFTn1400rpmc[key]=pd.concat(FFTn1400rpm[key]).groupby("Index").mean()
    FFTn2800rpmc[key]=pd.concat(FFTn2800rpm[key]).groupby("Index").mean()

# i=1
# plt.figure()
# for key in FFTn1400rpm.keys():
#     plt.subplot(2,3,i)
#     plt.title("FFT 1400 rpm - Valve Position open = "+key)
#     plt.stem(X, FFTn1400rpmc[key], markerfmt="")
#     plt.xlabel("Frequency in Hz")
#     plt.ylim(0,0.03)
#     plt.xlim(60,180)
#     i=i+1
# plt.show()

# i=1
# for key in FFTn2800rpm.keys():
#     plt.subplot(2,3,i)
#     plt.title("FFT 2800 rpm - Valve Position open = "+key)
#     plt.stem(X, FFTn2800rpmc[key], markerfmt="")
#     plt.xlabel("Frequency in Hz")
#     # plt.ylim(0,0.05)
#     plt.xlim(230,250)
#     i=i+1
# plt.show()

# Adding the Frequency information to the concatenated Data for checking
for key in FFTn1400rpm.keys():
    FFTn1400rpmc[key]["Frequency"] = X.tolist()
    FFTn2800rpmc[key]["Frequency"] = X.tolist()
    for i in range(len(FFTn1400rpm[key])):
        FFTn1400rpm[key][i]["Frequency"] = X.tolist()
    for i in range(len(FFTn2800rpm[key])):
        FFTn2800rpm[key][i]["Frequency"] = X.tolist()

#Concoatenating the remarkable values into a DataFrame
test_info = []
flap_pos = []
peak_rotation = []
peak_bladepass = []
v_rms = []
a_rms = []
for keys in FFTn1400rpm:
    for i in range(len(FFTn1400rpm[keys])):
        test_info.append(1400)
        flap_pos.append(int(keys.strip("%")))
        slice = FFTn1400rpm[keys][i][FFTn1400rpm[keys][i]["Frequency"].between(10,30)]
        peak_rotation.append(slice["Amplitude"].loc[slice["Amplitude"].idxmax()])
        
        slice = FFTn1400rpm[keys][i][FFTn1400rpm[keys][i]["Frequency"].between(110,130)]
        peak_bladepass.append(slice["Amplitude"].loc[slice["Amplitude"].idxmax()])
        
        v_rms.append(cal.v_rms(FFTn1400rpm[keys][i], 10, 30, X))
        a_rms.append(cal.a_rms(FFTn1400rpm[keys][i], 10, 30, X))
for keys in FFTn2800rpm:
    for i in range(len(FFTn2800rpm[keys])):
        test_info.append(2800)
        flap_pos.append(int(keys.strip("%")))
        
        slice = FFTn2800rpm[keys][i][FFTn2800rpm[keys][i]["Frequency"].between(30,40)]
        peak_rotation.append(slice["Amplitude"].loc[slice["Amplitude"].idxmax()])
        
        slice = FFTn2800rpm[keys][i][FFTn2800rpm[keys][i]["Frequency"].between(230,250)]
        peak_bladepass.append(slice["Amplitude"].loc[slice["Amplitude"].idxmax()])
        
        v_rms.append(cal.v_rms(FFTn2800rpm[keys][i], 10, 30, X))
        a_rms.append(cal.a_rms(FFTn2800rpm[keys][i], 10, 30, X))   
Data = pd.DataFrame({"test_info":test_info, "flap_pos_open":flap_pos, "peak_rotation":peak_rotation, "peak_bladepass":peak_bladepass, "v_rms":v_rms, "a_rms":a_rms})
Data.sort_values(by=["flap_pos_open","test_info"], inplace=True)
print(Data)

dataheader = []
for col in Data.columns:
    dataheader.append(col)
del dataheader[0:2]
"""Plotting the different remarkable values to see which can be used to differenciate"""
#1400rpm
i=1
plt.figure()
for info in dataheader:
    for info2 in dataheader:
        plt.subplot(4,4,i)
        plt.title(f"1400rpm")
        sns.scatterplot(Data[Data["test_info"]==1400], x=info2, y=info, hue="flap_pos_open")
        i+=1
plt.show()
#2800 rpm
i=1
plt.figure()
for info in dataheader:
    for info2 in dataheader:
        plt.subplot(4,4,i)
        plt.title(f"2800rpm")
        sns.scatterplot(Data[Data["test_info"]==2800], x=info2, y=info, hue="flap_pos_open")
        i+=1
plt.show()



y = Data[Data["test_info"]==1400]["flap_pos_open"]
X=Data[Data["test_info"]==1400][["peak_bladepass","a_rms"]]
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=0.25, random_state=0)        
model = GaussianNB()
model.fit(Xtrain,ytrain)
ypred = model.predict(Xtrain)
correct = accuracy_score(ytrain,ypred, normalize=False)
incorrect = np.size(ytrain)-correct
print(f"1400rpm-peak_bladepass/a_rms: Score=W{incorrect}-R{correct} accuracy={accuracy_score(ytrain,ypred):.2f}%")

y = Data[Data["test_info"]==2800]["flap_pos_open"]
X=Data[Data["test_info"]==2800][["peak_bladepass","a_rms"]]
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=0.25, random_state=0)        
model = GaussianNB()
model.fit(Xtrain,ytrain)
ypred = model.predict(Xtrain)
correct = accuracy_score(ytrain,ypred, normalize=False)
incorrect = np.size(ytrain)-correct
print(f"2800rpm-peak_bladepass/a_rms: Score=W{incorrect}-R{correct} accuracy={accuracy_score(ytrain,ypred):.2f}%")