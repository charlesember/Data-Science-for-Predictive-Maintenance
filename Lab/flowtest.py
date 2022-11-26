import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from scipy.fft import fft
import numpy as np

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

f_low = 35
f_high= 45
X_filt = X [X>f_low]
#Calculating the RMS
for keys in FFTn1400rpmc:
    RMSVal["FFTn1400rpm "+keys] = 2*FFTn1400rpmc[keys]["Amplitude"].to_numpy()
    RMSVal["FFTn1400rpm "+keys][1:] = RMSVal["FFTn1400rpm "+keys][1:]/np.sqrt(2)
    RMSVal["FFTn1400rpm "+keys] = RMSVal["FFTn1400rpm "+keys]*1.633/2
    RMSVal["FFTn1400rpm "+keys] = RMSVal["FFTn1400rpm "+keys][X>f_low]
    RMSVal["FFTn1400rpm "+keys] = RMSVal["FFTn1400rpm "+keys][X_filt<f_high]
    RMSVal["FFTn1400rpm "+keys] = np.sqrt(sum(RMSVal["FFTn1400rpm "+keys]**2))
    
    RMSVal["FFTn2800rpm "+keys] = 2*FFTn2800rpmc[keys]["Amplitude"].to_numpy()
    RMSVal["FFTn2800rpm "+keys][1:] = RMSVal["FFTn2800rpm "+keys][1:]/np.sqrt(2)
    RMSVal["FFTn2800rpm "+keys] = RMSVal["FFTn2800rpm "+keys]*1.633/2
    RMSVal["FFTn2800rpm "+keys] = RMSVal["FFTn2800rpm "+keys][X>f_low]
    RMSVal["FFTn2800rpm "+keys] = RMSVal["FFTn2800rpm "+keys][X_filt<f_high]
    RMSVal["FFTn2800rpm "+keys] = np.sqrt(sum(RMSVal["FFTn2800rpm "+keys]**2))

print("\nThe RMS Values are:")
print("Area: "+str(f_low)+"-"+str(f_high)+" Hz\n")
for keys in RMSVal:
    print(keys+" : "+str("{:.3f}".format(RMSVal[keys]))+" m/s^2")
print("\n")
    

i=1
plt.figure()
for key in FFTn1400rpm.keys():
    plt.subplot(2,3,i)
    plt.title("FFT 1400 rpm - Valve Position open = "+key)
    plt.stem(X, FFTn1400rpmc[key], markerfmt="")
    plt.xlabel("Frequency in Hz")
    plt.ylim(0,0.03)
    plt.xlim(100,200)
    i=i+1
plt.show()

i=1
for key in FFTn2800rpm.keys():
    plt.subplot(2,3,i)
    plt.title("FFT 2800 rpm - Valve Position open = "+key)
    plt.stem(X, FFTn2800rpmc[key], markerfmt="")
    plt.xlabel("Frequency in Hz")
    plt.ylim(0,0.05)
    plt.xlim(200,300)
    i=i+1
plt.show()

for key in FFTn1400rpm.keys():
    FFTn1400rpmc[key]["Frequency"] = X.tolist()
