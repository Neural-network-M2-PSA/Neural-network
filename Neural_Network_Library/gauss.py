"""
M2-PSA project
2020-2021
BROUILLARD Alizée, alizeebrouillard020198@gmail.com
BRUANT Quentin, quentinbruant92@gmail.com
GODINAUD Leila, leila.godinaud@gmail.com

"""

"""_______gauss_____"""
'''
This function return the  .....
'''

#Library
import numpy as np

def gauss(x,mu=0.25,sigma=0.05):
    return 1/(sigma* np.sqrt(2*np.pi)) * np.exp(-((x-mu)**2)/(2*sigma**2))