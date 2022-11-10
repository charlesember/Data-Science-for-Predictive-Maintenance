import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import *

#Reading in the FFT Spectrum
FFTcols = ["Frequency","Amplitude","B","C"]
FFT = []
FFT.append(pd.read_csv("./Data/Boottest/FFT/01_SPECTRUM_ ch1_ 2-3200Hz_ m_s^2_2022_11_04_15_28_42_275.csv", header=None, skiprows=1, sep=";", names=FFTcols))
FFT.append(pd.read_csv("./Data/Boottest/FFT/01_SPECTRUM_ ch1_ 2-3200Hz_ m_s^2_2022_11_04_15_30_08_347.csv", header=None, sep=";", skiprows=1, names=FFTcols))
FFT.append(pd.read_csv("./Data/Boottest/FFT/01_SPECTRUM_ ch1_ 2-3200Hz_ m_s^2_2022_11_04_15_31_10_723.csv", header=None, sep=";", skiprows=1, names=FFTcols))

for i in range(len(FFT[2])):
    if i%4 != 0:
        FFT[2].drop(i, inplace=True)

for ffts in FFT:
    ffts.drop(columns=["C","B"], inplace=True) #Dropping legacy columns
    # Making an Index Column to concatenate
    index=[]
    for i in range(len(ffts)):
        index.append(i)
    ffts.insert(0,"Index", index)
    ffts.dropna(inplace=True) #Dropping all Nan Values
    #print(ffts.describe()) #No nonNum Values detected

ffts_concat = pd.concat([FFT[0],FFT[1],FFT[2]]).groupby("Index").mean()

plt.figure
plt.stem(ffts_concat["Frequency"],ffts_concat["Amplitude"], markerfmt="")
plt.show()
plt.figure
plt.stem(FFT[0]["Frequency"], FFT[0]["Amplitude"], markerfmt="")
plt.show()