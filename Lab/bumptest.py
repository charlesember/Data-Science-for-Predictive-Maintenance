import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import *

#Reading in the FFT Spectrum
FFTcols = ["Frequency","Amplitude","B","C"]
FFT = []
FFT.append(pd.read_csv("./Data/BumpTest/FFT/01_SPECTRUM_ ch1_ 2-3200Hz_ m_s^2_2022_11_04_15_16_42_507.csv", header=None, skiprows=1, sep=";", names=FFTcols))
FFT.append(pd.read_csv("./Data/BumpTest/FFT/01_SPECTRUM_ ch1_ 2-3200Hz_ m_s^2_2022_11_04_15_18_30_286.csv", header=None, sep=";", skiprows=1, names=FFTcols))
FFT.append(pd.read_csv("./Data/BumpTest/FFT/01_SPECTRUM_ ch1_ 2-3200Hz_ m_s^2_2022_11_04_15_19_00_637.csv", header=None, sep=";", skiprows=1, names=FFTcols))

timecols = ["Time", "Amplitude","B","C"]
timesig = pd.read_csv("./Data/BumpTest/Times/01_TIME_ ch1_ 2-3200Hz_ m_s^2_2022_11_04_15_19_47_058.csv", header=None, sep=";", skiprows=1, names=timecols)
timesig.drop(columns=["B","C"], inplace=True)
timesig.dropna(inplace=True)

# Making an Index Column to concatenate
index=[]
for i in range(len(FFT[0])):
    index.append(i)

for ffts in FFT:
    ffts.insert(0,"Index", index)
    ffts.drop(columns=["C","B"], inplace=True) #Dropping legacy columns
    ffts.dropna(inplace=True) #Dropping all Nan Values
    #print(ffts.describe()) #No nonNum Values detected

ffts_concat = pd.concat([FFT[0],FFT[1],FFT[2]]).groupby("Index").mean()

plt.figure
plt.stem(ffts_concat["Frequency"],ffts_concat["Amplitude"], markerfmt="")
plt.show()
plt.figure()
plt.plot(timesig["Time"], timesig["Amplitude"])
plt.xlim(left=1.37, right=1.5)
plt.show()