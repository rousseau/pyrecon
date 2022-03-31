#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 10:09:43 2021

@author: mercier
"""
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from skimage.color import gray2rgb


def Histo(MSE):
    histoMse =  plt.hist(MSE,range=(min(MSE),max(MSE)),bins='auto')
    return histoMse

def show_slice(slices):  # definition de la fonction show_slice qui prend en paramètre une image
    # ""Function di display row of image slices""
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        # affiche l'image en niveau de gris (noir pour la valeur minimale et blanche pour la valeur maximale)
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


def plotsegment(Slice, pointImg, ok, nbpoint,ax,title=' ',mask=np.nan,index=np.nan,nbpointSlice=None):
    """
    The function plotSegment display a slice with it segment of intersection

    Inputs :
        
    slice : slice 
        type slice, contains all the necessary information about the slice, including data and mask
    pointImg : 3D array 
        point that delineate the segment of intersection
    ok : integer
        1 if there is an intersection, 0 else
    nbpoint : integer
        number of point in the segment (before applying the mask)
    title : string, OPTIONAL
        title of the image,  the defalut is ' '
    mask : 2D array,  OPTIONAL
        mask associated the the slice, The default is np.nan.
    index : 1D array, OPTIONAL
        index of the part of the segment including in the mask (cf commonProfil). The default is np.nan.
    nbpointSlice : interger, OPTION
        Number of points in the segment, after aplying the mask. The default is None.
    """
    sliceimage=Slice.get_slice().get_fdata()
    if ok < 1 or nbpoint==0:
        
        if ~np.isnan(mask).all():
            ax.imshow(sliceimage.T*mask,cmap='gray',origin='lower')
            ax.title(title)
        else:
           ax.imshow(sliceimage.T,cmap='gray',origin='lower')
           ax.title(title) 
    
    else:
        
        pointInterpol = np.zeros((2, nbpoint))
        pointInterpol[0, :] = np.linspace(pointImg[0, 0], pointImg[0, 1], nbpoint)
        pointInterpol[1, :] = np.linspace(pointImg[1, 0], pointImg[1, 1], nbpoint)
        
        if  ~np.isnan(mask).all():
           
            img_with_mask = nib.Nifti1Image(sliceimage * mask, Slice.get_slice().affine)
            pointInterpol = pointInterpol[:,index]
            nbpoint = pointInterpol.shape[1]
       
        ax.imshow(img_with_mask.get_fdata().squeeze().T, cmap="gray", origin="lower")
       
        for i in range(nbpoint):
            ax.plot(pointInterpol[0, i],pointInterpol[1, i], 'ro',markersize=1,label='%nbpoint')
         
        ax.text(0,0,"nbpoint : %d" %(nbpoint))
        if  nbpointSlice!=None:
            ax.text(0,0.1,"nbpointSlice : %d" %(nbpointSlice))
        ax.set_title(title)
    

def displayIntensityProfil(commonVal1,index1,commonVal2,index2,index):
    """
    The function display the intensity profil for two slices along the segment of intersection

    Inputs :
    commonVal1 : 1D array
        
    index1 : 1D array
        index of interest in the value commonVal1.
    commonVal2 : 1D array
        values of intensity along the segment
    index2 : 1D array
        index of interest in the value commonVal2
    index : 1D array
        union of the index of interest in the two array

    """
    print(index.size)
    if index.size == 0 : 
        return 0
    if index1.all()==False and ~index2.all()==False: 
        index1 = index2.copy()
    elif index2.all()==False and ~index1.all()==False:
        index2=index1.copy()

    
    plt.figure()
    indiceIndex1 = np.where(index1 == True)[0]
    size = np.shape(indiceIndex1)[0]
    #print(indiceIndex1)
    if(size>0):
        plt.axvline(x=indiceIndex1[0]-index[0],color='r')
        plt.axvline(x=indiceIndex1[size-1]-index[0],color='r')
        plt.plot(commonVal1,color='r')
            
    indiceIndex2 = np.where(index2 == True)[0]
    size = np.shape(indiceIndex2)[0]
    #print(indiceIndex2)
    if(size>0):
        plt.axvline(x=indiceIndex2[0]-index[0],color='b')
        plt.axvline(x=indiceIndex2[size-1]-index[0],color='b')
        plt.plot(commonVal2,color='b')
        plt.title('Intensity profil delineated by the mask')
    

def indexMse(gridError,gridNbpoint,numSlice):
    """
    The function computes the Mean Square Error between slice 1 and its orthogonal slices
    and return an array of the MSE between slice1 and each slice

    Inputs :
    girdError : 
        2D array, contains the SE between each pair of slices
    gridNbpoint : 
        2D array, contains the number of common point on the intersection between each pair of slices
    numSlice : 
        The slice of interest

    Ouptputs : 
    MSE :
        Array containing the MSE between slice1 and its orthogonal slices
    NBPOINT : 
        Array contaning the number of point in the union between each point

    """
    
    size,size=gridError.shape
    MSE = np.zeros(size)
    NBPOINT = np.zeros(size)
    for i_slice2 in range(size):
            newError = gridError[max(numSlice,i_slice2),min(numSlice,i_slice2)]
            commonPoint = gridNbpoint[max(numSlice,i_slice2),min(numSlice,i_slice2)]
            MSE[i_slice2] = newError
            NBPOINT[i_slice2] = commonPoint
    return MSE,NBPOINT 

def indexGlobalMse(gridError,gridNbpoint):
    """
    The function computes the Mean Square Error between each slices and its orthogonal slices 
    
    IInputs :
    girdError : 
        2D array, contains the SE between each pair of slices
    gridNbpoint : 
        2D array, contains the number of common point on the intersection between each pair of slices

    Ouptputs : 
    MSE_GLOB :
        Array containing the MSE between each slice and its orthogonal slices
    NBPOINT : 
        Array contaning the number of point in the union between each point

    """
    
    size,size=gridError.shape
    MSE_GLOB = np.zeros(size)
    NBPOINT_GLOB= np.zeros(size)
     
    for i_slice1 in range(size):
         errorSlice, nbpointSlice = indexMse(gridError,gridNbpoint,i_slice1)
         globalErrorSlice = sum(errorSlice)
         globalnbpointSlice = sum(nbpointSlice)
         NBPOINT_GLOB[i_slice1] = globalnbpointSlice
         MSE_GLOB[i_slice1] = globalErrorSlice
    

    return NBPOINT_GLOB,MSE_GLOB 
  

    
def indexDice(slice1,listSlice):
    """
    The function computes the DICE (intersection over union) between slice1 and its orthonal slices
    and return an array of the dice between slice1 and each slice.

    Inputs
    slice1 : 
        type slice, contains all the necessary information about the slice
    
    listSlice : 
        list of type slice, contains the images in the three orientations, axial,sagital and coronal

    Outputs :
    DICE : 
        Array containing the DICE between slice1 and its orthogonal slice

    """
    INTERSECTION = np.zeros(len(listSlice))
    UNION = np.zeros(len(listSlice))
    indice = 0
    for slice2 in listSlice:
        if slice1.get_orientation() != slice2.get_orientation(): #there is no intersection between slices and its orthogonal slices
            newIntersection,newUnion = DICElocal(slice1,slice2)
            INTERSECTION[indice]=newIntersection
            UNION[indice] = newUnion
            indice = indice+1
    return INTERSECTION,UNION

def indexGlobalDice(listSlice):
     INTERSECTION = np.zeros(len(listSlice))
     UNION = np.zeros(len(listSlice))
     indice=0
     for slice1 in listSlice:
         intersectionSlice, unionSlice = indexDice(slice1,listSlice)
         globalIntersectionSlice = sum(intersectionSlice)
         globalUnionSlice = sum(unionSlice)
         INTERSECTION[indice] = globalIntersectionSlice
         UNION[indice] = globalUnionSlice
         indice=indice+1
     return INTERSECTION,UNION  

