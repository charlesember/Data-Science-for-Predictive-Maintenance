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
n1400rpmc = {
    "0%": [],
    "25%": [],
    "50%": [],
    "75%": [],
    "100%": []
}
n2800rpmc = {
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
       
"""Concatenating the data with their mean value"""
for keys in n1400rpmc.keys():
    n1400rpmc[keys]=pd.concat(n1400rpm[keys]).groupby("Time",as_index=False).mean()
    n2800rpmc[keys]=pd.concat(n2800rpm[keys]).groupby("Time",as_index=False).mean()

"""Convert the dataframe into an numpy array"""
for keys in n1400rpmc.keys():
    n1400rpmc[keys]=n1400rpmc[keys].to_numpy()
    n2800rpmc[keys]=n2800rpmc[keys].to_numpy()

"""Making the FFT spectre"""
fs = 6400
N = np.size(n1400rpmc["0%"][:,1])
X = fs/N*np.arange(0,N//2)

for keys in FFTn1400rpm:
    FFTn1400rpm[keys] = fft(n1400rpmc[keys][:,1])
    FFTn1400rpm[keys] = np.abs(FFTn1400rpm[keys][0:N//2])/N
    FFTn1400rpm[keys][1:]=2*FFTn1400rpm[keys][1:]
    FFTn2800rpm[keys] = fft(n2800rpmc[keys][:,1])
    FFTn2800rpm[keys] = np.abs(FFTn2800rpm[keys][0:N//2])/N
    FFTn2800rpm[keys][1:]=2*FFTn2800rpm[keys][1:]

f_low = 10
f_high= 1000
X_filt = X [X>f_low]
#Calculating the RMS
for keys in FFTn1400rpm:
    RMSVal["FFTn1400rpm "+keys] = 2*FFTn1400rpm[keys]
    RMSVal["FFTn1400rpm "+keys][1:] = RMSVal["FFTn1400rpm "+keys][1:]/np.sqrt(2)
    RMSVal["FFTn1400rpm "+keys] = RMSVal["FFTn1400rpm "+keys]*1.633/2
    RMSVal["FFTn1400rpm "+keys] = RMSVal["FFTn1400rpm "+keys][X>f_low]
    RMSVal["FFTn1400rpm "+keys] = RMSVal["FFTn1400rpm "+keys][X_filt<f_high]
    RMSVal["FFTn1400rpm "+keys] = np.sqrt(sum(RMSVal["FFTn1400rpm "+keys]**2))
    
    RMSVal["FFTn2800rpm "+keys] = 2*FFTn2800rpm[keys]
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
    

# i=1
# plt.figure()
# for key in FFTn1400rpm.keys():
#     plt.subplot(2,3,i)
#     plt.title("Valve Position open = "+key)
#     plt.stem(X, FFTn1400rpm[key], markerfmt="")
#     plt.xlabel("Frequency in Hz")
#     plt.ylim(0,0.08)
#     i=i+1
# plt.show()

# i=1
# for key in FFTn2800rpm.keys():
#     plt.subplot(2,3,i)
#     plt.title("Valve Position open = "+key)
#     plt.stem(X, FFTn2800rpm[key], markerfmt="")
#     plt.xlabel("Frequency in Hz")
#     plt.ylim(0,0.8)
#     i=i+1
# plt.show()

