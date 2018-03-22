#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 14:17:23 2018

@author: qbe
"""
import numpy as np
import copy as cp  
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fft2, ifft2, fftshift

import Starlet2D as tp1
import BERTRAND_MIALON_tools_projet as to
import pyBSS as bss


####################################################################################################################""
#on charge les données
import scipy.io
mat = scipy.io.loadmat('Fourier_Measurements.mat')
y = mat['B']
mat = scipy.io.loadmat('Fourier_Sampling.mat')
M = mat['mask']
mat = scipy.io.loadmat('Noise_single_simulation.mat')
noise = mat['noise2']
mat =  scipy.io.loadmat('Input_FRG.mat')
Sreal = mat['frg_input']

plt.figure()
plt.title('source originelle 1', fontsize=18)
plt.imshow(Sreal[:,:,0], cmap='gray')

plt.figure()
plt.title('source originelle 2', fontsize=18)
plt.imshow(Sreal[:,:,1], cmap='gray')

plt.figure()
plt.title('masque', fontsize=18)
plt.imshow(M[:,:,5], cmap='gray')


####################################################################################################################"
#modèle de synthèse
def getGradf(c,w, mask, y, nLevel):
    x = tp1.Starlet_Backward2D(c, w)
    Fx = fft2(x, norm="ortho")-y
    res = mask * Fx
    res = ifft2(res, norm="ortho")
    res = np.real(res)
    cNew, wNew = tp1.Starlet_Forward2D(res, J=nLevel)
    return cNew, wNew

def FBS(Niter, x0, mask, y,noise, nLevel, Lambda, k=3, multiscale = False):
    nu = 1
    gamma = 1.9/nu
    theta = 2 - gamma * nu /2
    xOld = cp.copy(x0)
    (c, w ) = tp1.Starlet_Forward2D(x = xOld, J = nLevel)
    arrayLambdas = to.getDetectionLevels(noise, k, nLevel)
    print( arrayLambdas)
    for i in range(Niter):
        if (i%10 == 0):
            print("iteration " + str(i+1) + "/" + str(Niter) )
        (gradC, gradW) = getGradf(c, w, mask, y, nLevel)
        wHalf = w - gamma * gradW  
        c = c - gamma *gradC
        if  (multiscale == False):
            w = w + theta  * (to.softThrd(wHalf, gamma * Lambda) - w)
        else:
            w = w + theta  * (to.softThrdMultiScale(wHalf, gamma * arrayLambdas) - w)
        x = tp1.Starlet_Backward2D(c,w )
        print("error = ", to.error(x0, x))
    res = tp1.Starlet_Backward2D(c, w)
    return res


#####################################################################################################################
#débruitage canal par canal
Niter = 200
x0 = np.unif(0, 10, (256,256))
nLevel = 3
Lambda = 1
boolMultiscale = True

nObs = 10
for i in range(nObs):
    print("canal = ", str(i))
    y0 =y[:,:,i] 
    M0 = M[:,:,i]
    noise0 = noise[:,:,i]
    FBreconst = FBS(Niter, x0, M0, y0, noise0, nLevel, Lambda, multiscale=boolMultiscale)
    plt.figure()
    plt.title('image débruitée par FB', fontsize=18)
    plt.imshow(FBreconst, cmap='gray')
    path = 'ImagesProjet/synthesis_model_FBS_channel_' + str(i)  + "_" + str(Niter) + '.jpg'
    scipy.misc.imsave(path, FBreconst)
    