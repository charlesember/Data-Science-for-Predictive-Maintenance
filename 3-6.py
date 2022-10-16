from distutils.archive_util import make_archive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calc_wrms(Yvalues,f_low,f_high,picker):
    Y_filt = Yvalues[X > f_low]
    X_filt = X[X > f_low]
    Y_filt = Y_filt[X_filt < f_high]
    X_filt = X_filt[X_filt < f_high]
    if picker == "v_rms":
        v_rms = 1000/(2*np.pi)*np.sqrt(sum((Y_filt/X_filt)**2))
        return v_rms
    elif picker == "a_rms":
        a_rms = np.sqrt(sum(Y_filt**2))
        return a_rms
    else:
        return "enter a_rms or v_rms"
    
def calc_nrms(Yvalues,freq, range):
    Y_filt = Yvalues[X > (1-range)*freq]
    X_filt = X[X > (1-range)*freq]
    Y_filt = Y_filt[X_filt < (1+range)*freq]
    a_rms = np.sqrt(sum(Y_filt**2))
    return a_rms

def calc_crest(yvalues, N):
    a_peak = max([max(yvalues), abs(min(yvalues))])
    arms_tot = np.sqrt(sum(yvalues**2)/N)
    crest = a_peak/arms_tot
    return crest

data_Whole = []
data_Broken = []
data_Whole.append(pd.read_csv(f"./Daten3-6/S3.csv", sep=";", header=0))
data_Broken.append(pd.read_csv(f"./Daten3-6/S4.csv", sep=";", header=0))
for i in range(1,6):
    data_Whole.append(pd.read_csv(f"./Daten3-6/T{i}.csv", sep=";", header=0))   
for i in range(6,11):
    data_Broken.append(pd.read_csv(f"./Daten3-6/T{i}.csv", sep=";", header=0))

time = data_Whole[2].iloc[:,0].values
fs = round(1/(time[1]-time[0]))
N = np.size(data_Broken[1])
X = data_Whole[0].iloc[:,0].values
y_Whole = []
y_Broken = []
for i in range(len(data_Whole)):
    y_Whole.append(data_Whole[i].iloc[0:,1].values)
for i in range(len(data_Broken)):
    y_Broken.append(data_Broken[i].iloc[0:,1].values)

Wv_wrms = calc_wrms(y_Whole[0],10,1000, "v_rms")
Bv_wrms = calc_wrms(y_Broken[0],10,1000, "v_rms")
Wa_wrms = calc_wrms(y_Whole[0], 10, 2000, "a_rms")
Ba_wrms = calc_wrms(y_Broken[0],10, 2000, "a_rms")
Wa_nrms = calc_nrms(y_Whole[0], 1400, 0.05)
Ba_nrms = calc_nrms(y_Broken[0], 1400, 0.05)
Wcrest = calc_crest(y_Whole[1], N)
Bcrest = calc_crest(y_Broken[1], N)

print(f"Whole v_rms = {Wv_wrms:.3f} mm/s")
print(f"Broken v_rms = {Bv_wrms:.3f} mm/s")
print(f"Whole a_rms = {Wa_wrms:.3f} m/s^2")
print(f"Broken a_rms = {Ba_wrms:.3f} m/s^2")
print(f"Whole Narrowband a_rms = {Wa_nrms:.3f} mm/s^2")
print(f"Broken Narrowband a_rms = {Ba_nrms:.3f} mm/s^2")
print(f"Whole crest = {Wcrest:.3f}")
print(f"Broken crest = {Bcrest:.3f}")

plt.figure()
plt.subplot(2,2,1)
plt.stem(X,y_Whole[0], markerfmt="")
plt.subplot(2,2,2)
for i in range(1,len(y_Whole)):
    plt.plot(time,y_Whole[i])
plt.subplot(2,2,3)
plt.stem(X,y_Broken[0], markerfmt="")
plt.stem(X,y_Whole[0], markerfmt="")
plt.subplot(2,2,4)
for i in range(1,len(y_Broken)):
    plt.plot(time,y_Broken[i])
plt.show()

    
    