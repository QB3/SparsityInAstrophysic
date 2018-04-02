#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 20:58:25 2018

@author: qbe
"""
import numpy as np
import pyBSS as bss
import matplotlib.pyplot as plt
import scipy.io

import Starlet2D as tp1
from PIL import Image

##################################################################################################################
#chargement des données
n = 2 
m = 10
lenghtIm = 256
t = lenghtIm**2
X = np.zeros((m, lenghtIm, lenghtIm))

for i in range(m):
    path = "ImagesProjet/synthesis_model_FBS_channel_"+ str(i) +  "_1000.jpg"
    jpgfile = Image.open(path)
    X[i, :, :] = jpgfile


mat = scipy.io.loadmat('Input_FRG.mat')
Sreal = mat['frg_input']
Sreal = Sreal.T
#implémentation de PALM


def getGrad1(X, A, S):
    return np.dot(A.T, np.dot(A, S)-X) 

def getGrad2(X, A, S):
    return np.dot(np.dot(A, S)-X, S.T)

def getLip1(A):
    return np.linalg.norm(np.dot(A.T, A), ord = 'fro')

def getLip2(S):
    return np.linalg.norm(np.dot(S, S.T), ord = 'fro')


def softThrd(x, gamma):
    return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)

def Starlet_Forward2D_Multidim(S=0,h=[0.0625,0.25,0.375,0.25,0.0625],J=1,boption=3):
    (n, t)= S.shape
    tailleIm = int(np.sqrt(t))
    #on reshape les sources en image pour appliquer Starlet 2D
    newS = S.reshape((n, tailleIm, tailleIm))
    cMultidim = np.zeros((n, tailleIm, tailleIm))
    wMultidim = np.zeros((n, tailleIm, tailleIm, J))
    for i in range(n):
        cMultidim[i] , wMultidim[i,:,:,:]=  tp1.Starlet_Forward2D(newS[i, :, :], h,J,boption)
    return cMultidim, wMultidim

def Starlet_Backward2D_Multidim(cMultidim,wMultidim):
    (n, tailleIm, tailleIm, J) = wMultidim.shape
    SformatImage = np.zeros((n, tailleIm, tailleIm))
    for i in range(n):
        SformatImage[i,:,:] = tp1.Starlet_Backward2D(cMultidim[i], wMultidim[i,:,:,:])
    SformatLigne = SformatImage.reshape((n, tailleIm**2))
    return SformatLigne

def projOnBall(A):
    n, m = A.shape
    res = np.zeros((n,m))
    for i in range(m):
        Ci = A[:,i]
        norm = np.linalg.norm(Ci)
        if (norm <= 1):
            res[:,i] = Ci
        else:
            res[:,i] = Ci/norm
    return res
        
        
#attention il faut imposer des contraintes supplémentaires 
#A \in O_b : chacune des colonnes de A est normalisée en norme L2
#il faut donc rajouter un terme de contrainte : (d'où les 3 termes dans PALM)
def palm(nIter, X, A0, S0, gamma1 = 2, gamma2 = 2, Lambda = 1):
    import copy as cp    
    A = cp.copy(A0)
    S = cp.copy(S0)
    for i in range(nIter):
        if i%10 == 0:
            print("iteration ", i)
        ci = gamma2 * getLip2(S)
        grad2 = getGrad2(X, A, S)
        A = A - 1/ci * grad2
        A = projOnBall(A)
        di = gamma1 * getLip2(A)
        S = S - 1/di * getGrad1(X, A, S)
        cMultidim, wMultidim = Starlet_Forward2D_Multidim(S, J=2)
        wMultidim = softThrd(wMultidim, Lambda * di)
        S = Starlet_Backward2D_Multidim( cMultidim, wMultidim)
        print("evaluation bss = ", bss.Eval_BSS(A0, S0, A, S))
    return A, S


dimS = (n, t)
S0 = np.random.normal(0,1, dimS)
dimA = (m, n)
A0 = np.random.normal(0,1,dimA)
X = X.reshape((m, t))

############################################################################
#à réessayer avec plus d'itération
nIter = 500
A, S = palm(nIter, X, A0, S0, gamma1=1, gamma2=1, Lambda= 0.01)
A0=A
S0= S

##########################################################################
Spalm = S.reshape((n, lenghtIm, lenghtIm))

plt.figure()
plt.title('séparation par PALM', fontsize=18)
plt.imshow(-Spalm[0,:,:], cmap='gray')
path = 'ImagesProjet/PALM_1_'  + str(nIter) + '.jpg'
scipy.misc.imsave(path, -Spalm[0,:,:])

plt.figure()
plt.title('séparation par PALM', fontsize=18)
plt.imshow(-Spalm[1,:,:], cmap='gray')
path = 'ImagesProjet/PALM_2_'  + str(nIter) + '.jpg'
scipy.misc.imsave(path, -Spalm[1,:,:])

plt.figure()
plt.title('ground truth image', fontsize=18)
plt.imshow(Sreal[:,:,0], cmap='gray')
path = 'ImagesProjet/real_1_'  + str(nIter) + '.jpg'
scipy.misc.imsave(path, Sreal[:,:,0])


plt.figure()
plt.title('ground truth image', fontsize=18)
plt.imshow(Sreal[:,:,1], cmap='gray')
path = 'ImagesProjet/real_2_'  + str(nIter) + '.jpg'
scipy.misc.imsave(path, Sreal[:,:,1])

#################################################################
#analyse de la cohérence avec le modèle