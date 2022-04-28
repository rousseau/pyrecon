# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 13:59:55 2021

@author: mercier
"""


import nibabel as nib 
import numpy as np
from scipy.ndimage import map_coordinates
from sliceObject import SliceObject
from scipy.optimize import minimize
import random as rd
import time
from tools import createVolumesFromAlist, computeMeanRotation, computeMeanTranslation, ParametersFromRigidMatrix, rigidMatrix, rotationCenter, debug_meanMatrix
import os
import argparse
import numba
from numba import jit,njit
from multiprocessing import Pool
from functools import partial


@jit(nopython=True)
def intersectionLineBtw2Planes(M1,M2) : 
    """
    Compute the intersection line between two planes

    input : 
    M1 : 4x4 matrix
        3-D transformation that defines the first plane
         
    M2 : 4x4 matrix
        3-D transformtation that defines the second place
        

    Returns :
    coeff : 3x1 vector
        vector tangent to the line of intersection
    pt : 3x1 vector
        point on the line of intersection
    ok : integer
        1 if there is an intersection, else 0
    """  
    
    #normal vector to the 0xy plan
    n1 = np.cross(M1[0:3,0],M1[0:3,1]) 
    if np.linalg.norm(n1)<1e-6: #no division by 0, case M11 // M12 (normally that souldn't be the case)
        return np.array([np.float64(0.0)]),np.array([np.float64(0.0)]),np.array([np.float64(0.0)])
    n1norm = n1/np.linalg.norm(n1)
    n1 = n1norm
    t1 = M1[0:3,3]

    
    n2 = np.cross(M2[0:3,0],M2[0:3,1]) 
    if np.linalg.norm(n2)<1e-6: #no division by 0, case M21 // M22 (normally that souldn't be the case)
        return np.array([np.float64(0.0)]),np.array([np.float64(0.0)]),np.array([np.float64(0.0)])
    n2norm = n2/np.linalg.norm(n2)
    n2 = n2norm
    t2 = M2[0:3,3]

    
    alpha = np.ascontiguousarray(n1) @ np.ascontiguousarray(n2) #if the vector are colinear alpha will be equal to one (since n1 and n2 are normalized), can happend if we consider two parralel slice
    beta =  np.ascontiguousarray(n1) @ np.ascontiguousarray(t1)
    gamma = np.ascontiguousarray(n2) @ np.ascontiguousarray(t2)

    
    if abs((1 - alpha*alpha))<1e-6: #if the vector are colinear, there is no intersection
        return np.array([np.float64(0.0)]),np.array([np.float64(0.0)]),np.array([np.float64(0.0)])
    a = 1/(1 - alpha*alpha)
    g = a*(beta - alpha*gamma)
    h = a*(gamma - alpha*beta)

    #line equation
    coeff = np.cross(n1,n2)
    pt = g*n1 + h*n2

    return coeff, pt, np.array([np.float64(1.0)])
 
@jit(nopython=True)  
def intersectionSegment(sliceimage,M,coeff,pt):
    """
    Compute the segment of intersection between the line and the 2D slice
    
    input :
    sliceimage : Slice
        contains all the necessary information on the slice, including the 3D matrix transformation and the data from the slice)
    coeff : 3x1 vector
        vector tangent to the line of intersection
    pt :  3x1 vector
        point on the line 
    ok : integer
        1 if there is an intersection, else 0

    Output :
    lambdaPropo : 2 values of lambda which defines the intersection points on the line


    """
    
    #line equation into the image plan
    #M = sliceimage.get_transfo()
    
    Minv = np.linalg.inv(M)
    rinv = np.linalg.inv(M[0:3,0:3])
    n = np.ascontiguousarray(rinv) @ np.ascontiguousarray(coeff)
    pt = np.concatenate((pt,np.array([1])))
    ptimg = np.ascontiguousarray(Minv) @ np.ascontiguousarray(pt)
   
    a = -n[1]
    b = n[0]
    c = -(a*ptimg[0] + b*ptimg[1])
    
    
    #Intersection with the plan
    intersection = np.zeros((4,2)) #2 points of intersection of coordinates i,j,k 
    intersection[3,:] = np.ones((1,2))
    width = sliceimage.shape[0]-1
    height = sliceimage.shape[1]-1
 

    indice=0
    #The intersection on a corner are considered only once
    
    if (abs(a)>1e-10): #if a==0, the division by zeros in not possible, in this case we have only two intersection possible : 
           
        i=(-c/a); j=0;
        
        if  i >= 0 and i < width: #if y=0 x=-c/a  #the point (0,0) is considered here
            intersection[0,indice] =  i
            intersection[1,indice] =  j
            indice=indice+1
        
        i=((-c-b*height)/a); j=height
       
        if (i>0) and (i <= width) : #if y=height x=-(c-b* height)/a #the point  (width,height) is considered here
            intersection[0,indice] = i
            intersection[1,indice] = j
            indice=indice+1
        
         
    if (abs(b)>1e-10): #if b==0, the divistion by zeros in not possible, in this case we have only two intersection possible :
           
        i=0; j=(-c/b);
        
        if j>0 and  j <= height: #if x=0 y=-c/b #the point (0,heigth) is considered here
            intersection[0,indice] = i 
            intersection[1,indice] = j
            indice=indice+1
       
        i=width; j=(-c-a*width)/b
        
        if j>=0  and j<height: #if x=width y=(-c-a*width)/b  #the point (width,0) is considered here
            intersection[0,indice] = i
            intersection[1,indice] = j
            indice=indice+1

    
    if indice < 2 or indice > 2:
        return np.array([np.float64(0.0)]),np.array([np.float64(0.0)])
        
    #Compute the intersection point coordinates in the 3D space
    interw = np.zeros((4,2)) #2 points of intersection, with 3 coordinates x,y,z
    interw[3,:] = np.ones((1,2))
    interw = np.ascontiguousarray(M) @ np.ascontiguousarray(intersection) 
    
    interw[0:3,0] = interw[0:3,0] - pt[0:3]
    interw[0:3,1] = interw[0:3,1] - pt[0:3]
    
    squareNorm = np.ascontiguousarray(coeff) @ np.ascontiguousarray(coeff.transpose())
    lambdaPropo = np.ascontiguousarray((((1/squareNorm) * coeff.transpose())) @ np.ascontiguousarray(interw[0:3,:])) 

    
    return lambdaPropo,np.array([np.float64(1.0)])


    
@jit(nopython=True)    
def minLambda(lambdaPropo1,lambdaPropo2):
    
    """
    Compute the common segment between two images
    
    Inputs : 
        
    lambdaPropo1 : 2D vector
        2 values of lambda which represents the two intersection between the line and the slice1
    lambdaPropo2 : 2D vector
        2 values of lambda which represents the two intersection between the line and the slice2
    
    Outputs : 
        
    lambdaMin : 2D vector
        2 values of lamda which represents the common segment between the 2 slices
        
    """
    lambdaMin = np.zeros(2)
    lambdaMin[0] = min(min(lambdaPropo1),min(lambdaPropo2))
    lambdaMin[1] = max(max(lambdaPropo1),max(lambdaPropo2))

    
    return lambdaMin #Return 2 values of lambda that represent the common segment between the 2 slices

@jit(nopython=True)
def commonSegment(sliceimage1,M1,sliceimage2,M2,res):
    """
    Compute the coordinates of the two extremity points of the segment in the 2 image plans

    Inputs : 
    
    Slice1 : slice
        contains all the necessary information on the slice 1, including the transformation M into the 3D space and the information on the header
    Slice2: slice
        Contains all the necessary information on the slice 2, including the transformation M into the 3D space and the information on the header   
        
        
    Outputs : 
    
    pointImg1 : 3x2 matrix
        the extremites of the segment in the slice1 plan
    pointImg2 : 3x2 matrix
        the extremites of the segment in the slice1 plan
    nbpoint : integer
        number of points between the two extremities
    ok : interger
        1 if the common segment was computed well, 0 else    
        

    """
    
    #M1 = Slice1.get_transfo()
    #M2 = Slice2.get_transfo()
    
    coeff,pt,ok=intersectionLineBtw2Planes(M1,M2)
    ok=np.int(ok[0])

    
    if ok<1: #if there is no intersection lines (the 2 planes are parralel) it is useless to compute the intersectionSegment
        return np.zeros((2,2),dtype=np.float_),np.zeros((2,2),dtype=np.float_),np.zeros((2,2),dtype=np.float_),np.zeros((2,2),dtype=np.float_)
        
    #sliceimage1=Slice1.get_slice().get_fdata()
    lambdaPropo1,ok=intersectionSegment(sliceimage1,M1,coeff,pt) #if there is no intersection segment (the line of intersection is outisde of the image or on a corner), it useless to compute a common segment
    ok=np.int(ok[0])
   
    if ok<1:
        return np.zeros((2,2),dtype=np.float_),np.zeros((2,2),dtype=np.float_),np.zeros((2,2),dtype=np.float_),np.zeros((2,2),dtype=np.float_)
    
    #sliceimage2=Slice2.get_slice().get_fdata()
    lambdaPropo2,ok=intersectionSegment(sliceimage2,M2,coeff,pt)
    ok=np.int(ok[0])
    
    if ok<1:
        return np.zeros((2,2),dtype=np.float_),np.zeros((2,2),dtype=np.float_),np.zeros((2,2),dtype=np.float_),np.zeros((2,2),dtype=np.float_)

    lambdaMin = minLambda(lambdaPropo1,lambdaPropo2)
      
    if lambdaMin[0]==lambdaMin[1]: #the segment is nul, there is no intersection
        return np.zeros((2,2),dtype=np.float_),np.zeros((2,2),dtype=np.float_),np.zeros((2,2),dtype=np.float_),np.zeros((2,2),dtype=np.float_)
        
    point3D = np.zeros((3,2))
    
    point3D[0:3,0] = lambdaMin[0] * coeff + pt #Point corresponding to the value of lambda
    point3D[0:3,1] = lambdaMin[1] * coeff + pt
    
    point3D = np.concatenate((point3D,np.array([[1,1]])))
    
    pointImg1 = np.zeros((4,2))
    pointImg1[3,:] = np.ones((1,2))
    
    
    pointImg2 = np.zeros((4,2))
    pointImg2[3,:] = np.ones((1,2))
    
    
    pointImg1 = np.ascontiguousarray(np.linalg.inv(M1)) @ np.ascontiguousarray(point3D)

    pointImg2 = np.ascontiguousarray(np.linalg.inv(M2)) @ np.ascontiguousarray(point3D) 

    
    distance1 = np.linalg.norm(pointImg1[0:2,0] - pointImg1[0:2,1]) #distance between two points on the two images
    distance2 = np.linalg.norm(pointImg2[0:2,0] - pointImg2[0:2,1]) 


    #res = min(Slice1.get_slice().header.get_zooms())
      #the smaller resolution of a voxel
        
    if res<0: #probmem with the resolution of the image
        return np.zeros((2,2),dtype=np.float_),np.zeros((2,2),dtype=np.float_),np.zeros((2,2),dtype=np.float_),np.zeros((2,2),dtype=np.float_)
        
    if max(distance1,distance2)<1: #no pixel in commun
        return np.zeros((2,2),dtype=np.float_),np.zeros((2,2),dtype=np.float_),np.zeros((2,2),dtype=np.float_),np.zeros((2,2),dtype=np.float_)
       
    nbpoint = int(np.round(max(distance1,distance2)+1)/res) #choose the max distance and divide it by the smaller resolution 

    return pointImg1[0:2],pointImg2[0:2],(nbpoint)*np.ones((2,2),dtype=np.float_),np.ones((2,2),dtype=np.float_)


def sliceProfil(Slice,pointImg,nbpoint):
    """
    Interpol values on the segment to obtain the profil intensity

    Inputs : 
        
    Slice : slice
        type slice, contains all the necessary information about the slice, including data and mask 
    pointImg : 3x2 matrix
        the extremites of the segment in the slice plan (x1,x2;y1,y2)
    nbpoint : integer
        number of points between the two extremities

    Ouputs : 
    
    interpol : nbpointx1 vector
        values of intensity
    index : boolean nbpointx1 vector
        index of interest
    """
    if nbpoint == 0:
        return 0,0,0
    interpol= np.zeros(nbpoint)
    interpolMask = np.zeros(nbpoint)
    pointInterpol = np.zeros((3,nbpoint))
    pointInterpol[0,:] = np.linspace(pointImg[0,0],pointImg[0,1],nbpoint)
    pointInterpol[1,:] = np.linspace(pointImg[1,0],pointImg[1,1],nbpoint)
   
    
    mask = Slice.get_mask()
    map_coordinates(Slice.get_slice().get_fdata(), pointInterpol , output=interpol, order=1, mode='constant', cval=np.nan, prefilter=False)
    map_coordinates(mask, pointInterpol, output=interpolMask, order=0, mode='constant',cval=np.nan,prefilter=False)
    
    index =~np.isnan(interpol)*interpolMask>0
    #val_mask = interpol * interpolMask
    #index=val_mask>0
      
    return interpol,index,np.shape(index[index==True])[0]


def commonProfil(val1,index1,val2,index2,nbpoint,intersection='union'):
    """
    
    Compute the intensity of points of interest in the first slice or the second slice
    Inputs :
    
    val1 : nbpointx1 vector
        values of intensity in the first slice
    index1 : nbpointx1 vector
        values of interest in the first slice
    val2 : nbpointx1 vector
        values of intensity in the seconde slice
    index2 : nbpointx1 vector
        values of interest in the second slice

    Output :
    
    val1[index] : vector of the size of index
        values of interset in val1
    
    val2[index] : vector of the size of index
        values of interest in val2

    """
    if nbpoint==0:
        return 0,0,0
    
    valindex=np.linspace(0,nbpoint-1,nbpoint,dtype=int)
    if intersection=='union':
        index = index1+index2 
    
    elif intersection=='intersection':
        index = index1*index2
    
    else:
        print(intersection, "is not recognized, choose either 'itersection' or 'union'")
    
    index = valindex[index==True]
    commonVal1 = np.zeros(val1.shape[0])  - 2
    commonVal1[index1] = val1[index1]
    commonVal2 = np.zeros(val2.shape[0])  - 2
    commonVal2[index2] = val2[index2]
 
    return commonVal1[index],commonVal2[index],index
    

def loadSlice(img,mask,listSlice,orientation):
    """
    Create Z sliceObject from the slice and the mask, with z being the third dimension of the 3D image
    """
    if mask == None:
        mask = nib.Nifti1Image(np.ones(img.get_fdata().shape), img.affine)
    X,Y,Z = img.shape
    slice_img = np.zeros((X,Y,1))
    slice_mask = np.zeros((X,Y,1))
    for zi in range(Z): #Lecture des images
        slice_img[:,:,0] = img.get_fdata()[:,:,zi]
        slice_mask[:,:,0] = mask.get_fdata()[:,:,zi]
        if ~(np.all(slice_mask==0)):
            mz = np.eye(4)
            mz[2,3]= zi
            sliceaffine = img.affine @ mz
            nifti = nib.Nifti1Image(slice_img.copy(),sliceaffine)
            c_im1 = SliceObject(nifti,slice_mask.copy(),orientation)
            listSlice.append(c_im1)
            
        
    return listSlice


@jit(nopython=True)
def error(commonVal1,commonVal2):
    return np.sum((commonVal1 - commonVal2)**2)
    



def costLocal(slice1,slice2):
    """
    
    Compute the MSE between two slices

    Inputs : 

    slice1 : a slice of a 3D volume
        type slice, contains all the necessary information about the slice
    slice2 : slice
        type slice, contains all the necessary information about the slice

    Returns :
        
    newError : double
        square error between two slices
    commonPoint : integer
        number of common point on the intersection between the two slices
    MSEloc : double
        mean square error bteween two slices

    """
    sliceimage1=slice1.get_slice().get_fdata();sliceimage2=slice2.get_slice().get_fdata();res=min(slice1.get_slice().header.get_zooms())
    M1=slice1.get_transfo();M2=slice2.get_transfo()
    pointImg1,pointImg2,nbpoint,ok = commonSegment(sliceimage1,M1,sliceimage2,M2,res)
    ok=np.int(ok[0,0]); nbpoint=np.int(nbpoint[0,0])
    newError=0; commonPoint=0; inter=0; union=0
    
    if ok>0:
        val1,index1,nbpointSlice1=sliceProfil(slice1, pointImg1, nbpoint) 
        val2,index2,nbpointSlice2=sliceProfil(slice2, pointImg2, nbpoint)
        commonVal1,commonVal2,index=commonProfil(val1, index1, val2, index2,nbpoint)
        val1inter,val2inter,interpoint=commonProfil(val1, index1, val2, index2,nbpoint,'intersection')
        commonPoint=index.shape[0]
        newError=error(commonVal1,commonVal2)
        inter=interpoint.shape[0]
        union=commonPoint
    return newError,commonPoint,inter,union


def updateCostBetweenAllImageAndOne(indexSlice,listSlice,gridError,gridNbpoint,gridInter,gridUnion):
    """
    The function computes the MSE between slice1 and its orthogonal slices.

    Inputs:
    slice1 : slice
        type slice, contains all the necessary information about the slice
    listSlice : list of slices
        slices which represents our 3D volumes

    Returns:
    se : double
        square error between slice1 and its orthogonal slices
    nbpoint_mse : integer
        
    """
    slice1=listSlice[indexSlice]


        
    for i_slice2 in range(len(listSlice)):
        slice2=listSlice[i_slice2]
                
        if slice1.get_orientation() != slice2.get_orientation(): #there is no intersection between slices and its orthogonal slices

            if indexSlice > i_slice2 : 
                newError,commonPoint,inter,union=costLocal(slice1,slice2)
                gridError[indexSlice,i_slice2]=newError
                gridNbpoint[indexSlice,i_slice2]=commonPoint
                gridInter[indexSlice,i_slice2]=inter
                gridUnion[indexSlice,i_slice2]=union
            else:
                newError,commonPoint,inter,union=costLocal(slice2,slice1)
                gridError[i_slice2,indexSlice]=newError
                gridNbpoint[i_slice2,indexSlice]=commonPoint
                gridInter[i_slice2,indexSlice]=inter
                gridUnion[i_slice2,indexSlice]=union

    
    # #debug procedure :
    # debugGridError,debugGridNbpoint, debugGridInter, debugGridUnion = computeCostBetweenAll2Dimages(listSlice)
    # #print(debugGridError==gridCriterium)

    # if costFromMatrix(debugGridError,debugGridNbpoint) != costFromMatrix(gridError,gridNbpoint):
    #     print('error')
    #     return 0
    # if costFromMatrix(debugGridInter,debugGridUnion) != costFromMatrix(gridInter,gridUnion):
    #     print('error')
    #     return 0





def computeCostBetweenAll2Dimages(listSlice):
    """
    Computes the criterium between each slices 

    Inputs:
    listSlice : list of slices which represents the 3D volume

    Returns:
    globalError: double
        sum of square error between each slice
    totalNbpoint:
        sum of the number of common point on the intersection between each slices
        
    """
    nbSlice = len(listSlice)
    gridError=np.zeros((nbSlice,nbSlice))
    gridNbpoint=np.zeros((nbSlice,nbSlice))
    gridInter=np.zeros((nbSlice,nbSlice))
    gridUnion=np.zeros((nbSlice,nbSlice))
    i_slice1=0
    

    for i_slice1 in range(nbSlice):
        slice1=listSlice[i_slice1]
        for i_slice2 in range(nbSlice):
            slice2=listSlice[i_slice2]
            if (i_slice1 > i_slice2):
                if slice1.get_orientation() != slice2.get_orientation():
                        newError,commonPoint,inter,union=costLocal(slice1,slice2)
                        gridError[i_slice1,i_slice2]=newError
                        gridNbpoint[i_slice1,i_slice2]=commonPoint
                        gridInter[i_slice1,i_slice2]=inter
                        gridUnion[i_slice1,i_slice2]=union
                    
    return gridError,gridNbpoint,gridInter,gridUnion     





def cost_fct(x,i_slice, listSlice, gridError, gridNbpoint, gridInter, gridUnion, gridWeight,threshold): #slice and listSlice are constant parameters and x is variable
    """
    Compute the cost function. 
    
    Inputs :
        x : 6D array
            parameters of the rigid transformation. The first three parameters represent the rotation and the last three parameters represent the translation
    """
    slicei=listSlice[i_slice]
    slicei.set_parameters(x)
    updateCostBetweenAllImageAndOne(i_slice, listSlice, gridError, gridNbpoint, gridInter, gridUnion)
    #updateMatrixOfWeight(gridError,gridNbpoint,gridWeight,i_slice,threshold)
    #errorWithWeight = gridWeight * gridError
    tmpgrid = gridErrorWithWeight(gridWeight,gridError,i_slice,True)
    res = costFromMatrix(tmpgrid,gridNbpoint.copy())
    
    return res


#@jit(nopython=True)
def costFromMatrix(gridError,gridNbpoint):
    globalError = np.sum(gridError)
    globalNbpoint = np.sum(gridNbpoint)
    if globalNbpoint>0:
        MSE = globalError/globalNbpoint
    else:
        MSE=0
    return MSE
#def cost_fct()



def normalization(listSlice):
    """
    Normalized the images to a standard normal distribution

    Inputs
    image : 3D matrix
        The image we want to normalized

    Returns
    normalizedImg : 3D matrix
        The normalized image

    """
   
    mean_sum = 0
    n = 0
    std=0
    var = []
    for s in listSlice:  
        data = s.get_slice().get_fdata()*s.get_mask()
        X,Y,Z = data.shape
        for x in range(X):
            for y in range(Y):
                if data[x,y,0]!= 0:
                    mean_sum = mean_sum + data[x,y,0]
                    n = n + 1
                    var.append(data[x,y,0])
    mean = mean_sum/n
    std = np.sqrt((sum((np.array(var)-mean)*(np.array(var)-mean))/(n)))
    
    listSliceNorm = []
    for s in listSlice:
        slices = s.get_slice().get_fdata()*s.get_mask()
        X,Y,Z = slices.shape
        newSlice = np.zeros((X,Y,1))
        for x in range(X):
            for y in range(Y):
                if (slices[x,y,0] != 0):
                    newSlice[x,y,0] = (slices[x,y,0] - mean)/std
        newNifti = nib.Nifti1Image(newSlice, s.get_slice().affine)
        s.set_slice(newNifti)
        #data = s.get_slice().get_fdata()
        listSliceNorm.append(s)

    #dbug : 
    # mean_sum = 0
    # n = 0
    # std=0
    # var = []
    # for s in listSlice:  
    #     data = s.get_slice().get_fdata()*s.get_mask()
    #     X,Y,Z = data.shape
    #     for x in range(X):
    #         for y in range(Y):
    #             if data[x,y,0]!= 0:
    #                 mean_sum = mean_sum + data[x,y,0]
    #                 n = n + 1
    #                 var.append(data[x,y,0])
    # mean = mean_sum/n
    # std = np.sqrt((sum((np.array(var)-mean)*(np.array(var)-mean))/(n-1))) 
    return listSliceNorm
 

def global_optimization(listSlice):
    """
    Compute the optimised parameters for each slice

    Input
    
    listSlice : list of sliceObjects
        list of the 2D images of the fetus brains with the orientation information associated with each image

    Returns
    
    A list of Arrays. The arrays reprent the Evolution of the parameters that are use to validate the registration.
    The arrays are : the Evolution of the MSE, of the DICE, the square error, the number of points, the intersection, the union and the results transformation.

    """
    
    nbSlice=len(listSlice) #number of slices in the list
    EvolutionError=[] #initalisation of the result arrays
    EvolutionDice=[]
    EvolutionGridError=[] 
    EvolutionGridNbpoint=[]
    EvolutionGridInter=[]
    EvolutionGridUnion=[]
    EvolutionParameters=[] 
    Previous_parameters=np.zeros((6,nbSlice))
    EvolutionTransfo=[]
    
    #Initialisation :
    dicRes={}
    for i in range(nbSlice): #values of the parameters before starting the optimization
        slicei=listSlice[i]
        EvolutionParameters.extend(slicei.get_parameters())
        Previous_parameters[:,i]=slicei.get_parameters()
        EvolutionTransfo.extend(slicei.get_transfo())
        
    dicRes["evolutionparameters"] = EvolutionParameters
    dicRes["evolutiontransfo"] = EvolutionTransfo
    gridError,gridNbpoint,gridInter,gridUnion=computeCostBetweenAll2Dimages(listSlice) #error values before starting the optimization
    EvolutionGridError.extend(gridError.copy())
    dicRes["evolutiongriderror"] = EvolutionGridError
    EvolutionGridNbpoint.extend(gridNbpoint.copy())
    dicRes["evolutiongridnbpoint"] = EvolutionGridNbpoint
    EvolutionGridInter.extend(gridInter.copy())
    dicRes["evolutiongridinter"] = EvolutionGridInter
    EvolutionGridUnion.extend(gridUnion.copy())
    dicRes["evolutiongridunion"] = EvolutionGridUnion
    costMse=costFromMatrix(gridError, gridNbpoint)
    costDice=costFromMatrix(gridInter,gridUnion)
    
    EvolutionError.append(costMse)
    dicRes["evolutionerror"] = EvolutionError
    print('The MSE before optimization is :', costMse)
    EvolutionDice.append(costDice)
    dicRes["evolutiondice"] = EvolutionDice
    print('The DICE before optimisation is :', costDice)
    
    #Differents steps of the optimization
    #1 - Register all the slices together
    print('step 1 :', "all the slice are register together")
    gridError,gridNbpoint,gridInter,gridUnion,dicRes=allSlicesOptimisation(listSlice, gridError, gridNbpoint, gridInter, gridUnion, dicRes)
    #1 - 1) Compute of error threshold based on MSE, if the MSE of a slice is below this threshold, it is well register
    #2 - Register togeter the slices that are bad-gister
    #print('step 2 :', "the bad slices are register")
    #gridError,gridNbpoint,gridInter,gridUnion,dicRes=badSliceOptimisation(listSlice,Previous_parameters,gridError,gridNbpoint,gridInter,gridUnion,dicRes)
    #returns parameters used to validate the registration
    #dicRes=MeanOptimisation(listSlice,gridError,gridNbpoint,gridInter,gridUnion,dicRes)
    
    
    return dicRes
    

def SimplexOptimization(delta,x0,i_slice,listSlice,gridError,gridNbpoint,gridInter,gridUnion,gridWeight,initial_s,threshold):
    """
    Implementation of the simplex (Nealder - Mead) method for the problem
    
    Inputs : 
        delta : the size of the initial simplex
        x0 : the initial value of the initial simplex
        initial_s : an initial array for the simplex
        i_slice : index of the slice in listSlice 
        listSlice :  list of sliceObjects, list of the 2D images of the fetus brains with the orientation information associated with each image
        gridError : triangular matrix representing the square error between each slices
        gridNbpoint : triangular matrix representing the number of common point between each slices
        gridInter : triangular matrix representing the number of point on the intersection between each slices
        gridUnion : triangular matrix representing the number of point on the union between each slices
        gridWeight : binary matrix, each column represent one image. The matrix indicate if the slice is below or above the threshold
        threshold : scalar, if the MSE of a slice is above this value, it is well-register, if not, it is badly-register
        
    Outputs : 
        costMSE : global MSE after optimisation
        costDice : global DICE after optimisation
        
    """
    slicei=listSlice[i_slice]
    
    delta=5
    P0 = x0 #create the initial simplex
    P1 = x0 + np.array([delta,0,0,0,0,0])
    P2 = x0 + np.array([0,delta,0,0,0,0])
    P3 = x0 + np.array([0,0,delta,0,0,0])
    P4 = x0 + np.array([0,0,0,delta,0,0])
    P5 = x0 + np.array([0,0,0,0,delta,0])
    P6 = x0 + np.array([0,0,0,0,0,delta])
                                    
    initial_s[0,:]=P0
    initial_s[1,:]=P1
    initial_s[2,:]=P2
    initial_s[3,:]=P3
    initial_s[4,:]=P4
    initial_s[5,:]=P5
    initial_s[6,:]=P6
                                                
    X,Y = gridError.shape
    NM = minimize(cost_fct,x0,args=(i_slice,listSlice,gridError.copy(),gridNbpoint.copy(),gridInter.copy(),gridUnion.copy(),gridWeight.copy(),threshold),method='Nelder-Mead',options={"disp" : True,"maxiter" : 2000, "maxfev":1e6, "xatol" : 0.1, "fatol" : 1e-2, "initial_simplex" : initial_s , "adaptive" :  False})
    #optimisation of the cost function using the simplex method                                    
    
    x_opt = NM.x #best parameter obtains after registration
    slicei.set_parameters(x_opt)
    updateCostBetweenAllImageAndOne(i_slice, listSlice, gridError, gridNbpoint,gridInter,gridUnion) #updated matrix of cost
    costMse=costFromMatrix(gridError,gridNbpoint) #MSE after optimisation
    costDice=costFromMatrix(gridInter,gridUnion) #Dice after optimisation
    
    return costMse,costDice,gridError,gridNbpoint,gridInter,gridUnion,x_opt
    
def OptimizationThreshold(gridError,gridNbpoint):
    """
    Compute the threshold between well-register slices and bad-register slices
    
    Inputs 
    gridError : triangular matrix representing the square error between each slices
    gridNbpoint : triangular matrix representing the number of common point between each slices

    Ouptut 
    threshold : scalar, if the MSE of a slice is above this value, it is well-register, if not, it is badly-register
        
    """
    
    vectMse = np.zeros([gridError.shape[0]])
    
    for i in range(gridError.shape[0]): #compute between each slice and its orthogonal slices
        mse = sum(gridError[i,:]) + sum(gridError[:,i])
        point =  sum(gridNbpoint[i,:]) + sum(gridNbpoint[:,i])
        vectMse[i] = mse/point
                            
    valMse = np.median(vectMse[~np.isnan(vectMse)])
    
    threshold = 1.25*valMse #the therhold is 1.25x the median value (cf kim's article)
    
    return threshold


def updateResults(dicRes,gridError,gridNbpoint,gridInter,gridUnion,costMse,costDice,listSlice,nbSlice):
    
    dicRes["evolutionerror"].append(costMse)
    dicRes["evolutiondice"].append(costDice)
    dicRes["evolutiongriderror"].extend(gridError.copy())
    dicRes["evolutiongridnbpoint"].extend(gridNbpoint.copy())
    dicRes["evolutiongridinter"].extend(gridInter.copy())
    dicRes["evolutiongridunion"].extend(gridUnion.copy())
    
                      
    for i in range(nbSlice):
        slicei=listSlice[i]
        dicRes["evolutionparameters"].extend(slicei.get_parameters())
        dicRes["evolutiontransfo"].extend(slicei.get_transfo())
        
def allSlicesOptimisation(listSlice,gridError,gridNbpoint,gridInter,gridUnion,dicRes):
    """
    
    Register all slices together

    Input:
        
    listSlice : list of sliceObjects, list of the 2D images of the fetus brains with the orientation information associated with each image
    gridError : triangular matrix representing the square error between each slices
    gridNbpoint : triangular matrix representing the number of common point between each slices
    gridInter : triangular matrix representing the number of point on the intersection between each slices
    gridUnion : triangular matrix representing the number of point on the union between each slices
    EvolutionError : array of error at each iteration
    EvolutionDice : array of dice at each iteration
    EvolutionGridError : array of the evolution of the variable 'gridError' at each iteration
    EvolutionGridNbpoint : array of the evolution of the variable 'gridNbpoint' at each iteration
    EvolutionGridInter : array of the evolution of the variable 'gridInter' at each iteration
    EvolutionGridUnion : array of the evolution of variable 'gridUnion' at each iteration
    EvolutionParameters : array of the evolution parameters at each iteration
    EvolutionTransfo : array of the evolution of the transformation at each iteration

    Ouputs:
        
    gridError : triangular matrix representing the square error between each slices
    gridNbpoint : triangular matrix representing the number of common point between each slices
    gridInter : triangular matrix representing the number of point on the intersection between each slices
    gridUnion : triangular matrix representing the number of point on the union between each slices
    EvolutionError : array of error at each iteration
    EvolutionDice : array of dice at each iteration
    EvolutionGridError : array of the evolution of the variable 'gridError' at each iteration
    EvolutionGridNbpoint : array of the evolution of the variable 'gridNbpoint' at each iteration
    EvolutionGridInter : array of the evolution of the variable 'gridInter' at each iteration
    EvolutionGridUnion : array of the evolution of variable 'gridUnion' at each iteration
    EvolutionParameters : array of the evolution parameters at each iteration
    EvolutionTransfo : array of the evolution of the transformation at each iteration

    """
    
    delta=5 #maximum size of simplex
    initial_s = np.zeros((7,6))
    #vectd = np.linspace(delta,0.01,delta,dtype=int) #distinct size of simplex
    vectd = np.linspace(delta,1,delta,dtype=int)
    nbSlice = len(listSlice)

    images,mask=createVolumesFromAlist(listSlice)
    nbImages=len(images)
    nbSlice=len(listSlice)
    #proc=Process(target=SliceOptimisation(5, listSlice, gridError, gridNbpoint, gridInter, gridUnion, np.ones((nbSlice,nbSlice)), 10000, initial_s, EvolutionError, EvolutionDice, EvolutionGridError, EvolutionGridNbpoint, EvolutionGridInter, EvolutionGridUnion, EvolutionParameters, EvolutionTransfo))
    #procs.append(proc)
    #proc.start()
    
    maxIt=0

    for d in vectd: #optimize each slices for differents size of simplex
        
        
        maxIt=0
        print("d :", d,"maxIt", maxIt)
        previous_cost=38
        current_cost=costFromMatrix(gridError, gridNbpoint)
        while previous_cost>current_cost+0.001 and maxIt<50:
           
            finish=np.zeros(nbSlice)
            maxItSlice=0
            
            while finish.all()!=1 and maxItSlice<10:
                
                index_pre=0
                for n_image in range(nbImages):
                    start = time.time()
                    nbSliceImageN=len(images[n_image])
                    randomIndex= np.arange(nbSliceImageN)
                    #np.random.shuffle(randomIndex) 
                    index=randomIndex+index_pre

                    with Pool() as p:
                        tmpfun=partial(allSliceOptimisation1Image,d,listSlice,gridError,gridNbpoint,gridInter,gridUnion,np.ones((nbSlice,nbSlice)),10000,initial_s,finish) 
                        res=p.map(tmpfun,index)
                    
                    for i_slice in range(nbSliceImageN):
                        listSlice[randomIndex[i_slice]+index_pre].set_parameters(res[i_slice][0])
                        finish[randomIndex[i_slice]+index_pre]=res[i_slice][1]
                    gridError,gridNbpoint,gridInter,gridUnion=computeCostBetweenAll2Dimages(listSlice)
                    costMse=costFromMatrix(gridError,gridNbpoint)
    
                    costDice=costFromMatrix(gridInter,gridUnion)
                    index_pre=index_pre+len(images[n_image])
                    maxItSlice=maxItSlice+1
                    
            print('detla :',  d, 'iteration :', maxIt, 'final MSE :', costMse, 'final DICE :', costDice)
            maxIt=maxIt+1 
            previous_cost=current_cost
            current_cost=costFromMatrix(gridError, gridNbpoint)
            end = time.time()
            elapsed = end - start
            
            updateResults(dicRes,gridError,gridNbpoint,gridInter,gridUnion,costMse,costDice,listSlice,nbSlice)
            print('MSE: ', costMse)
            print('Dice: ', costDice) 
            print(f'Temps d\'exécution : {elapsed}')
            
    return gridError,gridNbpoint,gridInter,gridUnion,dicRes
    

def allSliceOptimisation1Image(d,listSlice,gridError,gridNbpoint,gridInter,gridUnion,gridWeight,threshold,initial_s,finish,nImage):    

    delta = d 
    slicei=listSlice[nImage]
    x0=slicei.get_parameters()
    previous_cost=oneSliceCost(gridError, gridNbpoint, nImage)
    finish_val=finish[nImage]
    if finish_val==0:
        costMse,costDice,gridError,gridNbpoint,gridInter,gridUnion,x_opt=SimplexOptimization(delta, x0, nImage, listSlice, gridError, gridNbpoint, gridInter, gridUnion, gridWeight, initial_s, threshold)
        current_cost=oneSliceCost(gridError, gridNbpoint, nImage)
        print('nbSlice :',nImage, 'costMse :', costMse, 'costDice :', costDice, 'previous_cost :', previous_cost, 'current_cost :', current_cost)
        if previous_cost<current_cost+0.001 or current_cost==np.inf:
            finish_val=1
    else:
        x_opt=x0

    return x_opt,finish_val          

def oneSliceCost(gridError,gridNbpoint,i_slice):
    error=sum(gridError[i_slice,:])+sum(gridError[:,i_slice])
    nbpoint=sum(gridNbpoint[i_slice,:])+sum(gridNbpoint[:,i_slice])
    if nbpoint==0:
        mse=0
    else:
        mse=error/nbpoint
    if np.isinf(mse):
        print('ALERT ERROR')
    return mse

def badSliceOptimisation1Image(d,listSlice,gridError,gridNbpoint,gridInter,gridUnion,gridWeight,threshold,initial_s,finish,nImage):
    

    delta = d 
    slicei=listSlice[nImage]
    x0=slicei.get_parameters()
    previous_cost=oneSliceCost(gridError, gridNbpoint, nImage)
    finish_val=finish[nImage]
    if gridWeight[0,nImage] == 0 and finish_val==0:
        costMse,costDice,gridError,gridNbpoint,gridInter,gridUnion,x_opt=SimplexOptimization(delta, x0, nImage, listSlice, gridError, gridNbpoint, gridInter, gridUnion, gridWeight, initial_s, threshold)
        print('nbSlice :',nImage, 'costMse :', costMse, 'costDice :', costDice)
        current_cost=oneSliceCost(gridError, gridNbpoint, nImage)
        if previous_cost<current_cost+0.001 or current_cost==np.inf:
            finish_val=1
    else:
        x_opt=x0
    return x_opt, finish_val 
   
def badSliceOptimisation(listSlice,PreviousParameters,gridError,gridNbpoint,gridInter,gridUnion,dicRes):
    """
    
    Register only the best slices together -> ie the slices that have a MSE below the threshold

    Input:
        
    listSlice : list of sliceObjects, list of the 2D images of the fetus brains with the orientation information associated with each image
    threshold : scalar, if the MSE of a slice is above this value, it is well-register, if not, it is badly-register
    gridError : triangular matrix representing the square error between each slices
    gridNbpoint : triangular matrix representing the number of common point between each slices
    gridInter : triangular matrix representing the number of point on the intersection between each slices
    gridUnion : triangular matrix representing the number of point on the union between each slices
    EvolutionError : array of error at each iteration
    EvolutionDice : array of dice at each iteration
    EvolutionGridError : array of the evolution of the variable 'gridError' at each iteration
    EvolutionGridNbpoint : array of the evolution of the variable 'gridNbpoint' at each iteration
    EvolutionGridInter : array of the evolution of the variable 'gridInter' at each iteration
    EvolutionGridUnion : array of the evolution of variable 'gridUnion' at each iteration
    EvolutionParameters : array of the evolution parameters at each iteration
    EvolutionTransfo : array of the evolution of the transformation at each iteration

    Ouputs:
        
    gridError : triangular matrix representing the square error between each slices
    gridNbpoint : triangular matrix representing the number of common point between each slices
    gridInter : triangular matrix representing the number of point on the intersection between each slices
    gridUnion : triangular matrix representing the number of point on the union between each slices
    EvolutionError : array of error at each iteration
    EvolutionDice : array of dice at each iteration
    EvolutionGridError : array of the evolution of the variable 'gridError' at each iteration
    EvolutionGridNbpoint : array of the evolution of the variable 'gridNbpoint' at each iteration
    EvolutionGridInter : array of the evolution of the variable 'gridInter' at each iteration
    EvolutionGridUnion : array of the evolution of variable 'gridUnion' at each iteration
    EvolutionParameters : array of the evolution parameters at each iteration
    EvolutionTransfo : array of the evolution of the transformation at each iteration

    """
    
    
    delta=10
    initial_s = np.zeros((7,6))
    vectd = np.linspace(delta,0.01,delta) 
    nbSlice=len(listSlice)
    
    threshold=OptimizationThreshold(gridError,gridNbpoint)
    print('threshold :',threshold)
    gridWeight=matrixOfWeight(gridError, gridNbpoint, threshold)
    
    # bad_register=0
    # for i_slice in range(nbSlice):
    #     okpre=False;okpost=False;nbSlice1=0;nbSlice2=0
    #     if gridWeight[0,i_slice]==0:
    #         for i in range(i_slice,0,-1):
    #             if gridWeight[0,i]==1 :
    #                 while (listSlice[i].get_orientation()==listSlice[i_slice].get_orientation()):
    #                     nbSlice1=i
    #                     okpre=True
    #                     break
    #         for j in range(i_slice,nbSlice):
    #            if gridWeight[0,j]==1:
    #                while (listSlice[j].get_orientation()==listSlice[i_slice].get_orientation()):
    #                     nbSlice2=j
    #                     okpost=True
    #                     break
    #     if okpre==True and okpost==True:
    #         Slice=listSlice[i_slice];Slice1=listSlice[nbSlice1];Slice2=listSlice[nbSlice2]
    #         pS1=Slice1.get_parameters(); pS2=Slice2.get_parameters() 
    #         MS1=rigidMatrix(pS1);MS2=rigidMatrix(pS2)
    #         RotS1=MS1[0:3,0:3];RotS2=MS2[0:3,0:3]
    #         print('RotS1 :',RotS1,'RotS2 :',RotS2)
    #         TransS1=MS1[0:3,3];TransS2=MS2[0:3,3]
    #         print('TransS1 :',TransS1)
    #         print('TransS2 :',TransS2)
    #         Rot=computeMeanRotation(RotS1,RotS2)
    #         Trans=computeMeanTranslation(TransS1,TransS2)
    #         M=np.zeros((4,4))
    #         M[0:3,0:3]=Rot;M[0:3,3]=Trans
    #         p=ParametersFromRigidMatrix(M)
    #         print('before :',Slice1.get_parameters(),'parameters : ',p,'After :',Slice2.get_parameters())
    #         Slice.set_parameters(p)
            

    #gridError,gridNbpoint,gridInter,gridUnion=computeCostBetweenAll2Dimages(listSlice)        
    # print('nb slices mal recalee :', bad_register)
    images,mask=createVolumesFromAlist(listSlice)
    nbImages=len(images)    
    # nbSlice=len(listSlice)
    # #procs=[]
    # #proc=Process(target=SliceOptimisation(5, listSlice, gridError, gridNbpoint, gridInter, gridUnion, np.ones((nbSlice,nbSlice)), 10000, initial_s, EvolutionError, EvolutionDice, EvolutionGridError, EvolutionGridNbpoint, EvolutionGridInter, EvolutionGridUnion, EvolutionParameters, EvolutionTransfo))
    # #procs.append(proc)
    # #proc.start()
    
    for d in vectd:
        maxIt=0
        previous_cost=2
        current_cost=costFromMatrix(gridError, gridNbpoint)
        while(previous_cost>current_cost+0.001) and maxIt<100:
            
            start=time.time()
            finish=np.zeros(nbSlice)
            maxItSlice=0
            
            while finish.all()!=1 and maxItSlice<100:
                
                index_pre=0
                for n_image in range(nbImages):
                    nbSliceImageN=len(images[n_image])
                    randomIndex= np.arange(nbSliceImageN)
                    #np.random.shuffle(randomIndex)  
                    index=randomIndex+index_pre
                 #for i in :
                    with Pool() as p:
                        tmpfun=partial(badSliceOptimisation1Image,d,listSlice,gridError,gridNbpoint,gridInter,gridUnion,gridWeight,threshold,initial_s,finish) 
                        res=p.map(tmpfun,index)
                    #x_opt=tmpfun(i_slice)
                    for i_slice in range(nbSliceImageN):
                        listSlice[randomIndex[i_slice]+index_pre].set_parameters(res[i_slice][0])
                        finish[randomIndex[i_slice]+index_pre]=res[i_slice][1]
                    #updateCostBetweenAllImageAndOne(i_slice, listSlice, gridError, gridNbpoint, gridInter, gridUnion)
                    gridError,gridNbpoint,gridInter,gridUnion=computeCostBetweenAll2Dimages(listSlice)
                    costMse=costFromMatrix(gridError,gridNbpoint)
                    costDice=costFromMatrix(gridInter,gridUnion)
                    index_pre=index_pre+len(images[n_image])
                    maxItSlice=maxItSlice+1
            previous_cost=current_cost
            current_cost=costMse
            maxIt=maxIt+1
            print('final MSE :', costMse, 'final DICE :', costDice)
           
            
            end = time.time()
            elapsed = end - start
                        
            print('MSE: ', costMse)
            print('Dice: ', costDice) 
            print(f'Temps d\'exécution : {elapsed}')
            
            updateResults(dicRes,gridError,gridNbpoint,gridInter,gridUnion,costMse,costDice,listSlice,nbSlice)
                        
            
    return gridError,gridNbpoint,gridInter,gridUnion,dicRes

def MeanOptimisation(listSlice,gridError,gridNbpoint,gridInter,gridUnion,dicRes):
    
    nbSlice=len(listSlice)
    threshold=OptimizationThreshold(gridError,gridNbpoint)
    print('threshold :',threshold)
    gridWeight=matrixOfWeight(gridError, gridNbpoint, threshold)
    
    for i_slice in range(nbSlice):
         print('i_slice :',i_slice)
         okpre=False;okpost=False;nbSlice1=0;nbSlice2=0;dist1=0;dist2=0
         slicei=listSlice[i_slice]
         if gridWeight[0,i_slice]==0:
             for i in range(i_slice,0,-1):
                 if listSlice[i].get_orientation()==listSlice[i_slice].get_orientation():
                     if gridWeight[0,i]==1 :
                         nbSlice1=i
                         dist1=np.abs(i_slice-nbSlice1)
                         print('dist1 :',dist1)
                         okpre=True
                         break
             for j in range(i_slice,nbSlice):
                if gridWeight[0,j]==1:
                    if listSlice[j].get_orientation()==listSlice[i_slice].get_orientation():
                         nbSlice2=j
                         dist2=np.abs(i_slice-nbSlice2)
                         print('dist2 :',dist2)
                         okpost=True
                         break
             if okpre==True and okpost==True:
                 Slice1=listSlice[nbSlice1];Slice2=listSlice[nbSlice2]
                 ps1=Slice1.get_parameters();ps2=Slice2.get_parameters();
                 MS1=rigidMatrix(ps1);MS2=rigidMatrix(ps2)
                 RotS1=MS1[0:3,0:3];RotS2=MS2[0:3,0:3]
                 TransS1=MS1[0:3,3];TransS2=MS2[0:3,3]
                 print('TransS1 :',TransS1)
                 print('TransS2 :',TransS2)
                 Rot=computeMeanRotation(RotS1,RotS2)
                 Trans=computeMeanTranslation(TransS1,dist1,TransS2,dist2)
                 print('Trans :',Trans)
                 Mtot=np.eye(4)
                 Mtot[0:3,0:3]=Rot;Mtot[0:3,3]=Trans
                 p=ParametersFromRigidMatrix(Mtot)
                 #debug_meanMatrix(Rot,Trans,p)
                 print('before :',Slice1.get_parameters(),'parameters : ',p,'After :',Slice2.get_parameters())
                 slicei.set_parameters(p)
             elif okpre==True and okpost==False:
                  Slice1=listSlice[nbSlice1]
                  ps1=Slice1.get_parameters()
                  slicei.set_parameters(ps1)
             elif okpost==True and okpre==False:
                  Slice2=listSlice[nbSlice2]
                  ps2=Slice2.get_parameters()
                  slicei.set_parameters(ps2)
             
             delta=5
             initial_s = np.zeros((7,6))
             vectd = np.linspace(delta,1,delta,dtype=int) 
            
             for d in vectd:
                 maxIt=0
                 previous_cost=2
                 current_cost=costFromMatrix(gridError, gridNbpoint)
                 
                 while(previous_cost>current_cost+0.001)and maxIt<50:
                     
                     x0=slicei.get_parameters()
                     costMse,costDice,gridError,gridNbpoint,gridInter,gridUnion,x_opt = SimplexOptimization(d,x0,i_slice,listSlice,gridError,gridNbpoint,gridInter,gridUnion,np.ones((nbSlice,nbSlice)),initial_s,10000)
                     slicei.set_parameters(x_opt)
                     maxIt=maxIt+1
                     previous_cost=current_cost
                     current_cost=costMse
                     print('delta :', d,'maxIt :', maxIt, 'mse : ',costMse, 'dice :', costDice)

     
    gridError,gridNbpoint,gridUnion,gridInter=computeCostBetweenAll2Dimages(listSlice)
    costMse=costFromMatrix(gridError,gridNbpoint)
    costDice=costFromMatrix(gridInter,gridUnion)
    updateResults(dicRes, gridError, gridNbpoint, gridInter, gridUnion, costMse, costDice, listSlice, nbSlice)
        
    return dicRes
# def worstSliceOptimization(listSlice,threshold,Previous_parameters,gridError,gridNbpoint,gridInter,gridUnion,EvolutionError,EvolutionDice,EvolutionGridError,EvolutionGridNbpoint,EvolutionGridInter,EvolutionGridUnion,EvolutionParameters,EvolutionTransfo):
#     """
    
#     Register only the worst slices together -> ie the slices that have a MSE above the threshold

#     Input:
        
#     listSlice : list of sliceObjects, list of the 2D images of the fetus brains with the orientation information associated with each image
#     threshold : scalar, if the MSE of a slice is above this value, it is well-register, if not, it is badly-register
#     Previous_parameters : parameters of the images before optimisation
#     gridError : triangular matrix representing the square error between each slices
#     gridNbpoint : triangular matrix representing the number of common point between each slices
#     gridInter : triangular matrix representing the number of point on the intersection between each slices
#     gridUnion : triangular matrix representing the number of point on the union between each slices
#     EvolutionError : array of error at each iteration
#     EvolutionDice : array of dice at each iteration
#     EvolutionGridError : array of the evolution of the variable 'gridError' at each iteration
#     EvolutionGridNbpoint : array of the evolution of the variable 'gridNbpoint' at each iteration
#     EvolutionGridInter : array of the evolution of the variable 'gridInter' at each iteration
#     EvolutionGridUnion : array of the evolution of variable 'gridUnion' at each iteration
#     EvolutionParameters : array of the evolution parameters at each iteration
#     EvolutionTransfo : array of the evolution of the transformation at each iteration

#     Ouputs:
        
#     gridError : triangular matrix representing the square error between each slices
#     gridNbpoint : triangular matrix representing the number of common point between each slices
#     gridInter : triangular matrix representing the number of point on the intersection between each slices
#     gridUnion : triangular matrix representing the number of point on the union between each slices
#     EvolutionError : array of error at each iteration
#     EvolutionDice : array of dice at each iteration
#     EvolutionGridError : array of the evolution of the variable 'gridError' at each iteration
#     EvolutionGridNbpoint : array of the evolution of the variable 'gridNbpoint' at each iteration
#     EvolutionGridInter : array of the evolution of the variable 'gridInter' at each iteration
#     EvolutionGridUnion : array of the evolution of variable 'gridUnion' at each iteration
#     EvolutionParameters : array of the evolution parameters at each iteration
#     EvolutionTransfo : array of the evolution of the transformation at each iteration

#     """
    
#     delta=5
#     vectMse = np.zeros([gridError.shape[0]])
#     initial_s = np.zeros((7,6))
#     vectd = np.linspace(delta,1,delta,dtype=int) 
#     nbSlice=len(listSlice)
    
#     gridWeight = matrixOfWeight(gridError, gridNbpoint, threshold)     
    
#     for i_slice in range(nbSlice):
#         slicei = listSlice[i_slice] 
#         slicei.set_parameters(Previous_parameters[:,i_slice])
#         if(gridWeight[0,i_slice] == 1):
#             for d in vectd:
#                 print('index slice: ',i_slice)
#                 print('Mse of the slice', vectMse[i_slice])
                                    
#                 x0=slicei.get_parameters()
#                 costMse,costDice=SimplexOptimization(delta, x0, i_slice, listSlice, gridError, gridNbpoint, gridInter, gridUnion, np.ones((nbSlice,nbSlice)), initial_s, 10000)
#                 print('costMse :', costMse, 'costDice :', costDice)
        
#     for i_slice in range(nbSlice):
#             slicei = listSlice[i_slice]
#             EvolutionParameters.extend(slicei.get_parameters())
#             EvolutionTransfo.extend(slicei.get_transfo())
            
#     EvolutionError.append(costMse)
#     EvolutionGridError.extend(gridError.copy())
#     EvolutionGridNbpoint.extend(gridNbpoint.copy())
#     EvolutionGridInter.extend(gridInter.copy())
#     EvolutionGridUnion.extend(gridUnion.copy())
#     EvolutionDice.append(costDice)            
    
#     return gridError,gridNbpoint,gridInter,gridUnion,EvolutionError,EvolutionDice,EvolutionGridError,EvolutionGridNbpoint,EvolutionGridInter,EvolutionGridUnion,EvolutionParameters,EvolutionTransfo

# def save_parameters(parameters2save,path):
#     listParameters = ['Anglex','Angley','Anglez','Translationx','Translationy','Translationz']
#     if not os.path.exists('path'):
#             os.makedirs('path')
#     for i in range(6):
#         if not os.path.exists('path/ %d'%(i)):
#                 os.makedirs('path/ %d' %(i))
#         file = 'path/ %d / %s .txt' %(i,listParameters[i])
#         np.savetxt(file,parameters2save[:,i,:])

    
def loadimages(fileImage,fileMask):
    
    im = nib.load(fileImage)
    if fileMask == None:
          fileMask = np.ones(im.get_fdata().shape)
          inmask = nib.Nifti1Image(fileMask,im.affine)
    else :
          inmask = nib.load(fileMask)
    return im,inmask
                
def matrixOfWeight(gridError,gridNbpoint,threshold):
    """
    Compute a binary matrix which represent if a slice is well-register of bad-register. Each column of the matrix represent a slice

    Inputs 
    gridError : triangular matrix representing the square error between each slices
    gridNbpoint : triangular matrix representing the number of common point between each slices
    threshold : scalar, if the MSE of a slice is above this value, it is well-register, if not, it is badly-register
        
    Ouptuts 
    Weight : The binary matrix

    """
    X,Y = gridError.shape
    mWeight = np.zeros((X,Y))
    valWeight = 0
    
    for i_slice in range(Y):
        valWeight = (sum(gridError[:,i_slice])+sum(gridError[i_slice,:]))/(sum(gridNbpoint[i_slice,:])+sum(gridNbpoint[:,i_slice]))
        
        mWeight[:,i_slice]=valWeight*np.ones(X)
        

    Weight = mWeight < threshold
    return Weight


def updateMatrixOfWeight(gridError,gridNbpoint,Weight,i_slice,threshold):
    """
    Updatate the matrix of weight for one slice    

    Inputs 
    gridError : triangular matrix representing the square error between each slices
    gridNbpoint : triangular matrix representing the number of common point between each slices
    threshold : scalar, if the MSE of a slice is above this value, it is well-register, if not, it is badly-register
        
    Ouptuts 
    Weight : The binary matrix

    """
    
    mWeight = Weight
    X,Y = gridError.shape
    
    valWeight=(sum(gridError[:,i_slice])+sum(gridError[i_slice,:]))/(sum(gridNbpoint[i_slice,:])+sum(gridNbpoint[:,i_slice]))
    
    weight = valWeight < threshold

    mWeight[:,i_slice]=weight*np.ones(X)
    
    return mWeight


    
    
def gridErrorWithWeight(Weight,gridError,i_slice,t):
    """
    Error matrix, adjusted with the gridWeight matrix
    """
    
    X,Y = gridError.shape
    gEwithWeight = np.zeros((X,Y))
    

    for i in range(X):
        for j in range(Y): 
            if i != i_slice:
                val1 = Weight[0,i]
            else:
                val1 = 1
            if j != i_slice:
                val2 = Weight[0,j]
            else:
                val2 = 1

        
            if i>j:
                if t == True:
                    gEwithWeight[i,j] = val1*val2*gridError[i,j]
                else:
                    gEwithWeight[i,j] = 1/((1/val1)+(1/val2))*gridError[i,j]

           
    return gEwithWeight
    
