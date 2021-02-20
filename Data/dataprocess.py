# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:14:12 2021

@author: Bruant Quentin
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import du DATAset
######################################################################################
######################################################################################

DATA = pd.read_csv('dataset.csv', sep=',', index_col=0)

#creation des different DATAset a normaliser ou non
######################################################################################
DATA_train = DATA[:int(len(DATA)/2)]
DATA_test = DATA[int(len(DATA)/2):int(3*len(DATA)/4)]
DATA_valid = DATA[int(3*len(DATA)/4):]


#fonction
######################################################################################
######################################################################################
def transform(X, bin_edges, X_cumul):
    X, B, F = np.array(X), np.array(bin_edges), np.array(X_cumul)
    R = np.zeros(len(X))
    for i in range(len(X)):
        for j in range(len(B)-1):
            if X[i] >= bin_edges[j] and X[i] < bin_edges[j+1]:
                R[i] = F[j]
    return R


#calcul des bornes pour Mbc
######################################################################################
M = np.max([DATA_train['Mbc']])
m = np.min([DATA_train['Mbc']])
#calcul des bormes pour m_KSpipi
######################################################################################
M_KSpipi = np.max([DATA_train['m_KSpipi']])
m_KSpipi = np.min([DATA_train['m_KSpipi']])


#normalisation
######################################################################################
Mbc_norm_train = (DATA_train['Mbc']-m)/np.abs(M-m)
Mbc_norm_test = (DATA_test['Mbc']-m)/np.abs(M-m)
Mbc_norm_valid = (DATA_valid['Mbc']-m)/np.abs(M-m)
Mbc_norm_train_signal = (DATA_train['Mbc'][DATA_train['isSignal'] == 1.]-m)/np.abs(M-m)

m_KSpipi_norm_train = (DATA_train['m_KSpipi']-m_KSpipi)/np.abs(M_KSpipi-m_KSpipi)
m_KSpipi_norm_test = (DATA_test['m_KSpipi']-m_KSpipi)/np.abs(M_KSpipi-m_KSpipi)
m_KSpipi_norm_valid = (DATA_valid['m_KSpipi']-m_KSpipi)/np.abs(M_KSpipi-m_KSpipi)
m_KSpipi_norm_train_signal = (DATA_train['m_KSpipi'][DATA_train['isSignal'] == 1.]-m_KSpipi)/np.abs(M_KSpipi-m_KSpipi)


#constructions des variables 
######################################################################################
######################################################################################

#Mbc
######################################################################################
Mbc_hist, bins_edges1 = np.histogram(Mbc_norm_train, bins=250, density=False)
Mbc_hist = np.insert(Mbc_hist, 0, 0)/len(Mbc_norm_train)
Mbc_signal_hist = np.histogram(Mbc_norm_train_signal, bins=250, density=False)[0]
Mbc_signal_hist = np.insert(Mbc_signal_hist, 0, 0)/len(Mbc_norm_train_signal)
Mbc_signal_cum = np.cumsum(Mbc_signal_hist)

R_Mbc_train = transform(Mbc_norm_train, bins_edges1, Mbc_signal_cum)
R_Mbc_test = transform(Mbc_norm_test, bins_edges1, Mbc_signal_cum)
R_Mbc_valid = transform(Mbc_norm_valid, bins_edges1, Mbc_signal_cum)


#m_KSpipi
######################################################################################
m_KSpipi_hist, bins_edges2 = np.histogram(m_KSpipi_norm_train, bins=250, density=False)
m_KSpipi_hist = np.insert(m_KSpipi_hist, 0, 0)/len(m_KSpipi_norm_train)
m_KSpipi_signal_hist = np.histogram(m_KSpipi_norm_train_signal, bins=250, density=False)[0]
m_KSpipi_signal_hist = np.insert(m_KSpipi_signal_hist, 0, 0)/len(m_KSpipi_norm_train_signal)
m_KSpipi_signal_cum = np.cumsum(m_KSpipi_signal_hist)

R_m_KSpipi_train = transform(m_KSpipi_norm_train, bins_edges2, m_KSpipi_signal_cum)
R_m_KSpipi_test = transform(m_KSpipi_norm_test, bins_edges2, m_KSpipi_signal_cum)
R_m_KSpipi_valid = transform(m_KSpipi_norm_valid, bins_edges2, m_KSpipi_signal_cum)


#cosTBz
#######################################################################################
cosTBz_train = DATA_train['cosTBz']
cosTBz_test = DATA_test['cosTBz']
cosTBz_valid = DATA_valid['cosTBz']
cosTBz_train_signal = DATA_train['cosTBz'][DATA_train['isSignal'] == 1.]

cosTBz_hist, bins_edges3 = np.histogram(cosTBz_train, bins=250, density=False)
cosTBz_hist = np.insert(cosTBz_hist, 0, 0)/len(cosTBz_train)
cosTBz_signal_hist = np.histogram(cosTBz_train_signal, bins=250, density=False)[0]
cosTBz_signal_hist = np.insert(cosTBz_signal_hist, 0, 0)/len(cosTBz_train_signal)
cosTBz_signal_cum = np.cumsum(cosTBz_signal_hist)

R_cosTBz_train = transform(cosTBz_train, bins_edges3, cosTBz_signal_cum)
R_cosTBz_test = transform(cosTBz_test, bins_edges3, cosTBz_signal_cum)
R_cosTBz_valid = transform(cosTBz_valid, bins_edges3, cosTBz_signal_cum)


#R2
#######################################################################################
R2_train = DATA_train['R2']
R2_test = DATA_test['R2']
R2_valid = DATA_valid['R2']
R2_train_signal = DATA_train['R2'][DATA_train['isSignal'] == 1.]

R2_hist, bins_edges4 = np.histogram(R2_train, bins=250, density=False)
R2_hist = np.insert(R2_hist, 0, 0)/len(R2_train)
R2_signal_hist = np.histogram(R2_train_signal, bins=250, density=False)[0]
R2_signal_hist = np.insert(R2_signal_hist, 0, 0)/len(R2_train_signal)
R2_signal_cum = np.cumsum(R2_signal_hist)

R_R2_train = transform(R2_train, bins_edges4, R2_signal_cum)
R_R2_test = transform(R2_test, bins_edges4, R2_signal_cum)
R_R2_valid = transform(R2_valid, bins_edges4, R2_signal_cum)


#chiProb
#######################################################################################
chiProb_train = DATA_train['chiProb']
chiProb_test = DATA_test['chiProb']
chiProb_valid = DATA_valid['chiProb']
chiProb_train_signal = DATA_train['chiProb'][DATA_train['isSignal'] == 1.]

chiProb_hist, bins_edges5 = np.histogram(chiProb_train, bins=250, density=False)
chiProb_hist = np.insert(chiProb_hist, 0, 0)/len(chiProb_train)
chiProb_signal_hist = np.histogram(chiProb_train_signal, bins=250, density=False)[0]
chiProb_signal_hist = np.insert(chiProb_signal_hist, 0, 0)/len(chiProb_train_signal)
chiProb_signal_cum = np.cumsum(chiProb_signal_hist)

R_chiProb_train = transform(chiProb_train, bins_edges5, chiProb_signal_cum)
R_chiProb_test = transform(chiProb_test, bins_edges5, chiProb_signal_cum)
R_chiProb_valid = transform(chiProb_valid, bins_edges5, chiProb_signal_cum)


#Ks_distChi
#######################################################################################
Ks_distChi_train = DATA_train['Ks_distChi']
Ks_distChi_test = DATA_test['Ks_distChi']
Ks_distChi_valid = DATA_valid['Ks_distChi']
Ks_distChi_train_signal = DATA_train['Ks_distChi'][DATA_train['isSignal'] == 1.]

Ks_distChi_hist, bins_edges6 = np.histogram(Ks_distChi_train, bins=250, density=False)
Ks_distChi_hist = np.insert(Ks_distChi_hist, 0, 0)/len(Ks_distChi_train)
Ks_distChi_signal_hist = np.histogram(Ks_distChi_train_signal, bins=250, density=False)[0]
Ks_distChi_signal_hist = np.insert(Ks_distChi_signal_hist, 0, 0)/len(Ks_distChi_train_signal)
Ks_distChi_signal_cum = np.cumsum(Ks_distChi_signal_hist)

R_Ks_distChi_train = transform(Ks_distChi_train, bins_edges6, Ks_distChi_signal_cum)
R_Ks_distChi_test = transform(Ks_distChi_test, bins_edges6, Ks_distChi_signal_cum)
R_Ks_distChi_valid = transform(Ks_distChi_valid, bins_edges6, Ks_distChi_signal_cum)


#plots
#######################################################################################
#######################################################################################

#Mbc
plt.figure()
plt.subplot(211)
plt.plot(bins_edges1, Mbc_hist)
plt.plot(bins_edges1, Mbc_signal_hist, 'r')
plt.subplot(212)
plt.plot(bins_edges1, Mbc_signal_cum, 'r')

plt.figure()
plt.hist(R_Mbc_train[DATA_train['isSignal'] == 1.], bins=15, alpha=0.6, color='r', label='signal')
plt.hist(R_Mbc_train[DATA_train['isSignal'] == 0.], bins=15, alpha=0.6, color='b', label='background')
plt.title('Beam constrained Mass (Mbc)')
plt.legend()

#m_KSpipi
plt.figure()
plt.subplot(211)
plt.plot(bins_edges2, m_KSpipi_hist)
plt.plot(bins_edges2, m_KSpipi_signal_hist, 'r')
plt.subplot(212)
plt.plot(bins_edges2, m_KSpipi_signal_cum, 'r')

plt.figure()
plt.hist(R_m_KSpipi_train[DATA_train['isSignal'] == 1.], bins=15, color='r', alpha=0.6, label='signal')
plt.hist(R_m_KSpipi_train[DATA_train['isSignal'] == 0.], bins=15, color='b', alpha=0.6, label='background')
plt.title('Renconstructed mass (m_KSpipi)')
plt.legend()

#cosTBz
plt.figure()
plt.subplot(211)
plt.plot(bins_edges3, cosTBz_hist)
plt.plot(bins_edges3, cosTBz_signal_hist, 'r')
plt.subplot(212)
plt.plot(bins_edges3, cosTBz_signal_cum, 'r')

plt.figure()
plt.hist(R_cosTBz_train[DATA_train['isSignal'] == 1.], bins=15, color='r', alpha=0.6, label='signal')
plt.hist(R_cosTBz_train[DATA_train['isSignal'] == 0.], bins=15, color='b', alpha=0.6, label='background')
#plt.title()
plt.legend()

#R2
plt.figure()
plt.subplot(211)
plt.plot(bins_edges4, R2_hist)
plt.plot(bins_edges4, R2_signal_hist, 'r')
plt.subplot(212)
plt.plot(bins_edges4, R2_signal_cum, 'r')

plt.figure()
plt.hist(R_R2_train[DATA_train['isSignal'] == 1.], bins=15, color='r', alpha=0.6, label='signal')
plt.hist(R_R2_train[DATA_train['isSignal'] == 0.], bins=15, color='b', alpha=0.6, label='background')
#plt.title()
plt.legend()

#chiProb
plt.figure()
plt.subplot(211)
plt.plot(bins_edges5, chiProb_hist)
plt.plot(bins_edges5, chiProb_signal_hist, 'r')
plt.subplot(212)
plt.plot(bins_edges5, chiProb_signal_cum, 'r')

plt.figure()
plt.hist(R_chiProb_train[DATA_train['isSignal'] == 1.], bins=15, color='r', alpha=0.6, label='signal')
plt.hist(R_chiProb_train[DATA_train['isSignal'] == 0.], bins=15, color='b', alpha=0.6, label='background')
#plt.title()
plt.legend()

#Ks_distChi
plt.figure()
plt.subplot(211)
plt.plot(bins_edges6, Ks_distChi_hist)
plt.plot(bins_edges6, Ks_distChi_signal_hist, 'r')
plt.subplot(212)
plt.plot(bins_edges6, Ks_distChi_signal_cum, 'r')

plt.figure()
plt.hist(R_Ks_distChi_train[DATA_train['isSignal'] == 1.], bins=15, color='r', alpha=0.6, label='signal')
plt.hist(R_Ks_distChi_train[DATA_train['isSignal'] == 0.], bins=15, color='b', alpha=0.6, label='background')
#plt.title()
plt.legend()

#remplacement des variables dans les DATAset correspondants
######################################################################################
######################################################################################
del DATA_train['Mbc']
DATA_train.insert(5, 'Mbc_norm', Mbc_norm_train)
del DATA_test['Mbc']
DATA_test.insert(5, 'Mbc_norm', Mbc_norm_test)
del DATA_valid['Mbc']
DATA_valid.insert(5, 'Mbc_norm', Mbc_norm_valid)
del DATA_train['m_KSpipi']
DATA_train.insert(4, 'm_KSpipi_norm', m_KSpipi_norm_train)
del DATA_test['m_KSpipi']
DATA_test.insert(4, 'm_KSpipi_norm', m_KSpipi_norm_test)
del DATA_valid['m_KSpipi']
DATA_valid.insert(4, 'm_KSpipi_norm', m_KSpipi_norm_valid)


#creation des dossier csv
###########################################
DATA_train.to_csv('DATA_train.csv')
DATA_test.to_csv('DATA_test.csv')
DATA_valid.to_csv('DATA_valid.csv')




#remplacement des variables facilitant la différenciation signal/background
######################################################################################
######################################################################################

#Mbc
del DATA_train['Mbc_norm']
del DATA_test['Mbc_norm']
del DATA_valid['Mbc_norm']

DATA_train.insert(5, 'R_Mbc', R_Mbc_train)
DATA_test.insert(5, 'R_Mbc', R_Mbc_test)
DATA_valid.insert(5, 'R_Mbc', R_Mbc_valid)

#m_KSpipi
del DATA_train['m_KSpipi_norm']
del DATA_test['m_KSpipi_norm']
del DATA_valid['m_KSpipi_norm']

DATA_train.insert(4, 'R_m_KSpipi', R_m_KSpipi_train)
DATA_test.insert(4, 'R_m_KSpipi', R_m_KSpipi_test)
DATA_valid.insert(4, 'R_m_KSpipi', R_m_KSpipi_valid)

#cosTBz
del DATA_train['cosTBz']
del DATA_test['cosTBz']
del DATA_valid['cosTBz']

DATA_train.insert(0, 'R_cosTBz', R_cosTBz_train)
DATA_test.insert(0, 'R_cosTBz', R_cosTBz_test)
DATA_valid.insert(0, 'R_cosTBz', R_cosTBz_valid)

#R2
del DATA_train['R2']
del DATA_test['R2']
del DATA_valid['R2']

DATA_train.insert(1, 'R_R2', R_R2_train)
DATA_test.insert(1, 'R_R2', R_R2_test)
DATA_valid.insert(1, 'R_R2', R_R2_valid)

#chiProb
del DATA_train['chiProb']
del DATA_test['chiProb']
del DATA_valid['chiProb']

DATA_train.insert(2, 'R_chiProb', R_chiProb_train)
DATA_test.insert(2, 'R_chiProb', R_chiProb_test)
DATA_valid.insert(2, 'R_chiProb', R_chiProb_valid)

#Ks_distChi
del DATA_train['Ks_distChi']
del DATA_test['Ks_distChi']
del DATA_valid['Ks_distChi']

DATA_train.insert(3, 'R_Ks_distChi', R_Ks_distChi_train)
DATA_test.insert(3, 'R_Ks_distChi', R_Ks_distChi_test)
DATA_valid.insert(3, 'R_Ks_distChi', R_Ks_distChi_valid)


#creation des dossier csv
###########################################
DATA_train.to_csv('DATA_train_fin.csv')
DATA_test.to_csv('DATA_test_fin.csv')
DATA_valid.to_csv('DATA_valid_fin.csv')






#DATA_train[DATA_train['isSignal'] == 1].hist('Mbc_norm', bins=250)


#print (DATA_train, DATA_test, DATA_valid, M, m)
