#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 10:00:47 2021

@author: mercier
"""
import numpy as np


def rotationCenter(mask,sliceaffine):
    """
    
    Compute the barycentre
    
    Inputs :
    mask : 2D image
        binary image which indicates the position of the brain

    Outputs : 
    centerw : 2xD vector
   
    """    
    X,Y = mask.shape
    
    center = np.zeros(2)

    somme_x = 0
    somme_y = 0
    nbpoint = 0
    
    for i in range(X):
        for j in range(Y):
                if mask[i,j] == 1:
                    somme_x = somme_x + i
                    somme_y = somme_y + j
                    nbpoint = nbpoint + 1
    center[0] = int(somme_x/nbpoint)
    center[1] = int(somme_y/nbpoint) 
    
    centerw = np.concatenate((center,np.array([0,1])))
    centerw = sliceaffine @ centerw 
    
    
    return centerw


def rigidMatrix(parameters):
    """
    Compute the rigidMatrix with 6 parameters. The first three parameters correspond to the rotation and the last three to the translation.

    Inputs : 
    parameters : The parameters of the rigid transformation, the first three parameters correspond to the rotation and the last three to the translation

    Outputs : 
    rigide : 4x4 matrix
    The translation matrix in homogenous coordinates

    """
   
    #convert angles into radiant
    gamma = np.pi*(parameters[0]/180.0)
    beta = np.pi*(parameters[1]/180.0)
    alpha = np.pi*(parameters[2]/180.0)
    
    # Rotalpha = np.eyes(4)
    # Rotalpha[1,1] = np.cos(alpha)
    # Rotalpha[1,2] = - np.sin(alpha)
    # Rotalpha[2,1] = np.sin(alpha)
    # Rotalpha[2,2] = np.cos(alpha)
    
    # Rotbeta = np.eyes(4)
    # Rotbeta[0,0] = np.cos(beta)
    # Rotbeta[0,2] = np.sin(beta)
    # Rotbeta[2,0] = -np.sin(beta)
    # Rotbeta[2,2] = np.cos(beta)
    
    
    # Rotgamma = np.eyes(4)
    # Rotgamma[0,0] = np.cos(gamma)
    # Rotgamma[0,1] = -np.sin(gamma)
    # Rotgamma[1,0] = np.sin(gamma)
    # Rotgamma[1,1] = np.cos(gamma)
    
    # trans = np.zeros(4)
    # trans[0:3,3] = translation
    
    # rigide = (Rotalpha * Rotbeta * Rotgamma) + trans 
    
    rigide = np.eye(4)
    rigide[0:3,3] = parameters[3:6]
    
    cosa=np.cos(alpha)
    cosg=np.cos(gamma)
    cosb=np.cos(beta)
    sina=np.sin(alpha)
    sing=np.sin(gamma)
    sinb=np.sin(beta)
    
    
    #rotation matrix, rotation around the axe x, y and k
    rigide[0,0] = cosa*cosb
    rigide[1,0] = cosa*sinb*sing-sina*cosg
    rigide[2,0] = cosa*sinb*cosg+sina*sing
    rigide[0,1] = sina*cosb
    rigide[1,1] = sina*sinb*sing+cosa*cosg
    rigide[2,1] = sina*sinb*cosg-cosa*sing
    rigide[0,2] = -sinb
    rigide[1,2] = cosb*sing
    rigide[2,2] = cosb*cosg
    
    return rigide