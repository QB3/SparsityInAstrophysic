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

path_to_pictures = '/Users/gregoire/Desktop/mva/parcimonie astrophysique/data'

####################################################################################################################""
#on charge les données
import scipy.io
(y, M, noise) = to.load_data


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
#x0 = np.unif(0, 10, (256,256))
nLevel = 3
Lambda = 1
boolMultiscale = True

nObs = 10

def function():
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

    return

def read_images(path):

    images = []

    for i in range(10):
        impath = path + '/%d.png' % i
        images.append(plt.imread(impath).ravel())

    return np.stack(images)

#séparation des sources avec FICA ou GMCA
def perform_source_separation(data, method):
    '''
    Comme précisé dans l'énoncé, on suppose que deux sources sont responsables des signaux mesurés. Cette méthode
    n'utilise pas le prior que l'on a sur la structure de A (exponentielle avec un facteur négatif pour une channel
    et un facteur positif pour l'autre.
    '''

    prior = [[20, 0],
             [18, 2],
             [16, 4],
             [14, 6],
             [12, 8],
             [10, 10],
             [8, 12],
             [6, 14],
             [4, 16],
             [200000000000000000, 18]]

    if method == 'prior_fica':
        A, S = bss.alt_Perform_FastICA(data, 2, prior)
        return A,S
    elif method == 'fica':
        A, S = bss.Perform_FastICA(data, 2)
        return A,S
    elif method == 'gmca':
        A, S, PinvA = bss.Perform_GMCA(data, 2)
        return A,S, PinvA
    else:
        raise('Wrong method, please choose either fica or gmca')

def display_source_separation(data):
    '''
    This function displays the two separated sources for both FICA and GMCA.
    '''

    models = ['fica', 'gmca']
    for model in models:
        if model == 'gmca':
            A, S, PinvA = perform_source_separation(data, model)
        elif model == 'fica':
            A, S = perform_source_separation(data, model)
        else:
            raise('Wrong method')
        for k in range(2):
            plt.imshow(S[k].reshape(256, 256))
            plt.show()
    return

def check_A(A):
    x = np.arange(0, len(A))
    for k in range(2):
        plt.scatter(x,np.log(np.abs((A[:,k]))))
    plt.show()
    return

if __name__ == '__main__':

    data = read_images(path_to_pictures)
    print(data.shape)
    A, S = perform_source_separation(data, 'prior_fica')
    print(A)
    check_A(A)