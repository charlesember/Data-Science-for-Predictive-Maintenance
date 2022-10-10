import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

fs = 1000
tmeas = 5
f = 220

t = np.arange(0,tmeas,1/fs)
y = np.sin(f*2*np.pi*t)
plt.figure
plt.plot(t,y)
plt.xlim([0,0.1])
sd.play(y,fs)
plt.show()


