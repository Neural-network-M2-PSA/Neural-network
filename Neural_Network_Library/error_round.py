"""
M2-PSA project
2020-2021
BROUILLARD Alizée, alizeebrouillard020198@gmail.com
BRUANT Quentin, quentinbruant92@gmail.com
GODINAUD Leila, leila.godinaud@gmail.com

"""

"""_______error_round_____"""
'''
This function returns the percentage of false answers with a round of the prediction to the unit
'''

# Library
import numpy as np

def error_round(y_prediction, y_target):
    y_prediction = np.round(y_prediction)
    y_target = np.round(y_target)
    nbr_error = y_target[y_prediction != y_target].size
    return nbr_error* 100/y_target.size
