import numpy as np

def error_round(y_prediction, y_actual):
    ''' pourcentage de réponse juste à 0.4 pres
    '''
    diff = y_prediction - y_actual
    diff_true = diff[np.abs(diff) <= 0.4 ]
    return 100 - diff_true.size*100/diff.size