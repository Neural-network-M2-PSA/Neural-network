"""
M2-PSA project
2020-2021
BROUILLARD Alizée, alizeebrouillard020198@gmail.com
BRUANT Quentin, quentinbruant92@gmail.com
GODINAUD Leila, leila.godinaud@gmail.com

"""

"""_______error_round_____"""
'''
This function return the percentage of true answer within 0.4
'''

#Library
import numpy as np

def error_round(y_prediction, y_actual):
    diff = y_prediction - y_actual
    diff_true = diff[np.abs(diff) <= 0.4 ]
    return 100 - diff_true.size*100/diff.size