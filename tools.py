#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 10:00:47 2021

@author: mercier
"""
import numpy as np



def rotationCenter(mask):
    """
    
    Compute the barycentre
    
    Inputs :
    mask : 2D image
        binary image which indicates the position of the brain

    Outputs : 
    centerw : 2xD vector
   
    """    
    shape = mask.shape
    X = shape[0]
    Y = shape[1]
    center = np.zeros(2)

    somme_x = 0
    somme_y = 0
    nbpoint = 0
    
    for i in range(X):
        for j in range(Y):
                if mask[i,j] > 0:
                    somme_x = somme_x + i
                    somme_y = somme_y + j
                    nbpoint = nbpoint + 1
    if nbpoint == 0: #in case there is no mask in the image, we consider the centre of rotation to be the center of the image
        center[0]= int(X/2)
        center[1]= int(Y/2)
    else:
        center[0] = int(somme_x/nbpoint)
        center[1] = int(somme_y/nbpoint) 
    
    centerw = np.concatenate((center,np.array([0,1])))
    #centerw = sliceaffine @ centerw 
    
    
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

    
    rigide = np.eye(4)
    rigide[0:3,3] = parameters[3:6]
    
    cosg=np.cos(gamma)
    cosa=np.cos(alpha)
    cosb=np.cos(beta)
    sing=np.sin(gamma)
    sina=np.sin(alpha)
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


def ParametersFromRigidMatrix(rigidMatrix):
    
    p=np.zeros(6)
    
    p[3]=rigidMatrix[0,3]
    p[4]=rigidMatrix[1,3]
    p[5]=rigidMatrix[2,3]
    
    beta=np.arcsin(-rigidMatrix[0,2])
    gamma=np.arctan2(rigidMatrix[1,2]/np.cos(beta),rigidMatrix[2,2]/np.cos(beta))
    alpha=np.arctan2(rigidMatrix[0,1]/np.cos(beta),rigidMatrix[0,0]/np.cos(beta))
    p[0]=(180.0*gamma)/np.pi
    p[1]=(180.0*beta)/np.pi
    p[2]=(180.0*alpha)/np.pi
    
    return p

#useful function for optimization shemes
def computeAllErrorsFromGrid(gridError,gridNbpoint):
    nbSlice,nbSlice = gridError.shape
    ArrayError = np.zeros(nbSlice)
    var_error = 0
    var_nbpoint = 0
    for i_slice in range(nbSlice):
        var_error = sum(gridError[:,i_slice]) + sum(gridError[:,i_slice])
        var_nbpoint = sum((gridNbpoint[i_slice,:])) + sum(gridNbpoint[:,i_slice])
        ArrayError[i_slice] = var_error/var_nbpoint 
    return ArrayError



def transfoMoy(M1,nbSlice1,M2,nbSlice2,nbSlice,Slice):
    
    a=np.abs(nbSlice-nbSlice1)
    print(nbSlice)
    print(nbSlice1)
    print('a=',a)
    b=np.abs(nbSlice-nbSlice2)
    print('b=',b)
    MtotB=(b*M1+a*M2)/(a+b)
    slicemask=Slice.get_mask()
    affine=Slice.get_slice().affine
    rotC = rotationCenter(slicemask) #Compute the barycenter of the image
    centerMatrix = np.eye(4)
    invcenterMatrix = np.eye(4)
    center = affine @ rotC
    centerMatrix[0:3,3] = -center[0:3]
    invcenterMatrix[0:3,3] = +center[0:3]
    MB=np.linalg.inv(invcenterMatrix)@MtotB@np.linalg.inv(affine)@np.linalg.inv(centerMatrix)
    return MB     
                