import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

fs = 1000
tmeas = 5
f1 = 220.32
t = np.arange(0,tmeas,1/fs)
y1 = np.sin(f1*2*np.pi*t)

N = np.size(y1)
X = fs/N*np.arange(0,N//2)
Y1 = fft(y1)
Y1 = np.abs(Y1[0:N//2])/N
Y1[1:] = 2*Y1[1:]

winb = np.hanning(N)
wins = np.hanning(np.size(Y1))
y1_win = winb*y1
Y1_win = wins*Y1

Y1_win = fft(y1_win)
Y1_win = np.abs(Y1_win[0:N//2])/N
Y1_win[1:] = 2*Y1_win[1:]

plt.figure
plt.subplot(2,2,1)
plt.plot(t,y1)
plt.xlabel("Time in s")
plt.title("Without Hann")

plt.subplot(2,2,2)
plt.stem(X,Y1, markerfmt="")
plt.xlabel("Frequency in Hz")
plt.title("Without Hann")

plt.subplot(2,2,3)
plt.plot(t,y1_win)
plt.xlabel("Time in s")
plt.title("With Hann")

plt.subplot(2,2,4)
plt.stem(X,Y1_win, markerfmt="")
plt.xlabel("Frequency in Hz")
plt.title("With Hann")
plt.show()



