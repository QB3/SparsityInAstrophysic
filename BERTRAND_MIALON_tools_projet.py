#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 10:49:34 2018

@author: qbe
"""
import numpy as np
import copy as cp  
import Starlet2D as tp1

#########################################################################################################################
#je recopie ici quelques fonctions du TP2
def softThrd(x, gamma):
    return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)

def hardThrd(x, gamma):
    bool0 = x > -gamma
    bool1 = x < gamma
    bool2 = np.logical_and(bool0, bool1)
    res = x * np.logical_not(bool2)
    return res

def softThrdMultiScale(x, listGamma):
    copyX = cp.copy(x)
    d = x.shape[2]
    for i in range(d):
        copyX[:,:,i] = softThrd(copyX[:,:,i] , listGamma[i])
    return copyX

def getSigmaMAD(w0):
    lho = 1.48826
    median = np.median(w0)
    sigmaMAD = lho * np.median(np.abs(w0 - median))
    return sigmaMAD

def getDetectionLevels(y, k, nbLevels):
    (c,w) = tp1.Starlet_Forward2D(x = y, J=nbLevels)
    res = np.zeros(nbLevels)
    for i in range(nbLevels):
        res[i] = k*getSigmaMAD(w[:,:,i])
    return res

def error(xStar, xHat):
    return np.linalg.norm(xStar - xHat) / np.linalg.norm(xStar)
