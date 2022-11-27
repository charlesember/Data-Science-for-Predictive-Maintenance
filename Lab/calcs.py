import numpy as np
import pandas as pd

def a_rms(Dataframe, xmin, xmax, X):
    X_filt = X [X>xmin]
    a_rms = 2*Dataframe["Amplitude"].to_numpy()
    a_rms[1:] = a_rms[1:]/np.sqrt(2)
    a_rms = a_rms*1.633/2
    a_rms = a_rms[X>xmin]
    a_rms = a_rms[X_filt<xmax]
    a_rms = np.sqrt(sum(a_rms**2))
    return a_rms

def v_rms(Dataframe, xmin, xmax, X):
    X_filt = X[X > xmin]
    
    v_rms = 2*Dataframe["Amplitude"].to_numpy()
    v_rms[1:] = v_rms[1:]/np.sqrt(2)
    v_rms = v_rms*1.633/2
    
    v_rms = v_rms[X > xmin]
    v_rms = v_rms[X_filt < xmax]
    X_filt = X_filt[X_filt < xmax]
    v_rms = 1000/(2*np.pi)*np.sqrt(sum((v_rms/X_filt)**2))
    return v_rms