import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

fs = 1000
tmeas = 5
f1 = 220.32
f2 = 220
t = np.arange(0,tmeas,1/fs)
y1 = np.sin(f1*2*np.pi*t)
y2 = np.sin(f2*2*np.pi*t)

N = np.size(y1)
X = fs/N*np.arange(0,N//2)
Y1 = fft(y1)
Y1 = np.abs(Y1[0:N//2])/N
Y1[1:] = 2*Y1[1:]

N = np.size(y2)
X2 = fs/N*np.arange(0,N//2)
Y2 = fft(y2)
Y2 = np.abs(Y2[0:N//2])/N
Y2[1:] = 2*Y2[1:]

plt.figure
plt.subplot(2,2,1)
plt.title("220.23 Hz")
plt.plot(t,y1)
plt.xlabel("Time in s")
plt.xlim(0,0.1)

plt.subplot(2,2,2)
plt.title("220.23 Hz")
plt.stem(X,Y1, markerfmt="")
plt.xlabel("Frequency in Hz")

plt.subplot(2,2,3)
plt.title("220 Hz")
plt.plot(t,y2)
plt.xlabel("Time in s")
plt.xlim(0,0.1)

plt.subplot(2,2,4)
plt.title("220 Hz")
plt.stem(X,Y2, markerfmt="")
plt.xlabel("Frequency in Hz")
plt.show()



