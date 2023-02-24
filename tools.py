#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 10:00:47 2021

@author: mercier
"""
import numpy as np
from numpy.linalg import eig,inv
from scipy.linalg import expm
from scipy.ndimage.filters import gaussian_filter
from nibabel import Nifti1Image



def denoising(listSlice,sigma):
    """
    
    Apply a gaussian filter to each slices in in a list
    
    Input

    listSlice : list wich contains slices from all stacks
    sigma : parameters from the gaussian filter

    Returns

    blurlist : A new list of slices with all slices blurred

    """
    #Initialization
    blurlist=[]
    
    for i_slice in range(len(listSlice)):
        slicei=listSlice[i_slice].copy()
        imgi=slicei.get_slice().get_fdata()
        imgi_modified = gaussian_filter(imgi,sigma=sigma) #applied gaussian filter to the image of the slice
        newSlice=Nifti1Image(imgi_modified, slicei.get_slice().affine) #create a new slice based on the modified image
        slicei.set_slice(newSlice)
        blurlist.append(slicei)
    return blurlist


def debug_meanMatrix(R_mean,Trans,parameters):
    """
    function to debug the men of rigid matrix

    """
    np.real(R_mean)
    res=np.zeros((4,4));res[3,3]=1
    res[0:3,0:3]=R_mean;res[0:3,3]=Trans
    
    r_matrix=rigidMatrix(parameters)
    print('res',res)
    print('r_mean',R_mean)
    
    equal=np.all(res==r_matrix)
    print(equal)
    return equal

def log_cplx(x):
    res = np.log(abs(x)) + np.angle(x)*1j
    return  res


def computeMeanRotation(R1,dist1,R2,dist2):
    """
    function to compute the mean of rotations, taking into account the distance from the two rotation 
    
    """
    M = R2 @ inv(R1)
    d,v = eig(M)
    tmp = log_cplx(d)
    A = v @ np.diag(tmp) @ inv(v)
    #R_mean= expm(A*dist1/(dist1+dist2))2 @ (np.exp(dist2/(dist1+dist2))*R1)
    R_mean=expm(A*dist1/(dist1+dist2)) @ R1
    R_mean=np.real(R_mean)
    
    return R_mean

def computeMeanTranslation(T1,dist1,T2,dist2):
    """
    function to compute the mean of translation
    """
    T_mean=((dist2) * T1 + (dist1) * T2)/(dist1 + dist2)
    return T_mean

def rotationCenter(mask):
    """
    
    Compute the barycentre
    
    Inputs :
    mask : 2D image
        binary image which indicates the position of the brain

    Outputs : 
    centerw : 2xD vector
   
    """    
    index = np.where(mask>0)
    center = np.sum(index,axis=1)/(np.sum(mask))
    centerw = np.concatenate((center[0:2],np.array([0,1])))
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
   
    """
    Find parameters associated with a rigidMatrix (3 parameters for rotation and 3 parameters for transaltion)
    """
    
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

  

def createVolumesFromAlist(listSlice):
   
    """
    re-create the differents original stacks of the list (ex : Axial, Sagittal, Coronal)
    """
    
    orientation = []; images=[]; mask=[]
    for s in listSlice:
        s_or = s.get_orientation()#s.get_index_image()
        if s_or in orientation:
            index_orientation = orientation.index(s_or)
            images[index_orientation].append(s)
            mask[index_orientation].append(s.get_mask())
        else:
            orientation.append(s_or)
            images.append([])
            mask.append([])
            index_orientation = orientation.index(s_or)
            images[index_orientation].append(s)
            mask[index_orientation].append(s.get_mask())
                
    return images, mask


def image_center(bin_image):
    index = np.where(bin_image>0)
    res = np.sum(index,axis=1)/(np.sum(bin_image))
    return res[0:3]

def center_image_2_ref(bin_image,affine_image,center_ref,affine_ref):
    
    matrix_2_ref = np.eye(4,4)
    center_world = affine_ref @ np.concatenate((center_ref,np.array([1])))
    print(center_world)
    matrix_2_ref[0:3,3] = center_world[0:3]
    
    return matrix_2_ref
    
    
    # mWeight = np.zeros((X,Y))



