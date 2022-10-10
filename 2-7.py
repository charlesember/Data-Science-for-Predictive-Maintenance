import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

fs = 2000
fb = 1000
tmeas = 5
f1, A1 = 220.32, 0.8
f2, A2 = 652, 0.95

ta = np.arange(0,tmeas,1/fs)
tb = np.arange(0,tmeas,1/fb)
y1 = A1*np.sin(f1*2*np.pi*ta)
y2 = A2*np.sin(f2*2*np.pi*ta)
y3 = A1*np.sin(f1*2*np.pi*tb)
y4 = A2*np.sin(f2*2*np.pi*tb)


plt.figure
plt.subplot(2,2,1)
plt.plot(ta,y1)
plt.xlim(0, 0.5)
plt.xlabel("Time in s")
plt.title("220 Hz, 2kHz")

plt.subplot(2,2,2)
plt.plot(ta,y2)
plt.xlabel("Time in s")
plt.xlim(0, 0.5)
plt.title("652 Hz, 2kHz")

plt.subplot(2,2,3)
plt.plot(tb,y3)
plt.xlim(0, 0.5)
plt.xlabel("Time in s")
plt.title("220 Hz, 1kHz")

plt.subplot(2,2,4)
plt.plot(tb,y4)
plt.xlabel("Time in s")
plt.xlim(0, 0.5)
plt.title("652 Hz, 1kHz")
plt.show()



