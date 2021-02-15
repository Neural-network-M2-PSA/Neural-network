import numpy as np

def gauss(x,mu=0.25,sigma=0.05):
    return 1/(sigma* np.sqrt(2*np.pi)) * np.exp(-((x-mu)**2)/(2*sigma**2))