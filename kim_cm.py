#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 13:59:55 2021

@author: mercier
"""


import nibabel as nib 
import numpy as np
import matplotlib.pyplot as plt 
from scipy.ndimage import map_coordinates
from sliceObject import SliceObject
from display import plotsegment,displayIntensityProfil
#from display import show_slice,plotsegment



##Chargement des images
img1 = nib.load('../donnee/export_chloe/sub-0259_ses-0307/sub-0259_ses-0307_t2_haste_te183_ax_6.nii.gz')
img2 = nib.load('../donnee/export_chloe/sub-0259_ses-0307/sub-0259_ses-0307_t2_haste_te183_cor_7.nii.gz')
img3 = nib.load('../donnee/export_chloe/sub-0259_ses-0307/sub-0259_ses-0307_t2_haste_te183_sag_8.nii.gz')
masque1 = nib.load('../donnee/export_chloe/sub-0259_ses-0307/brain_masks/sub-0259_ses-0307_t2_haste_te183_ax_6.nii.gz')
masque2 = nib.load('../donnee/export_chloe/sub-0259_ses-0307/brain_masks/sub-0259_ses-0307_t2_haste_te183_cor_7.nii.gz')
masque3 = nib.load('../donnee/export_chloe/sub-0259_ses-0307/brain_masks/sub-0259_ses-0307_t2_haste_te183_sag_8.nii.gz')


    

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
        return 0,0,0
    n1norm = n1/np.linalg.norm(n1)
    n1 = n1norm
    t1 = M1[0:3,3]

    
    n2 = np.cross(M2[0:3,0],M2[0:3,1]) 
    if np.linalg.norm(n2)<1e-6: #no division by 0, case M21 // M22 (normally that souldn't be the case)
        return 0,0,0
    n2norm = n2/np.linalg.norm(n2)
    n2 = n2norm
    t2 = M2[0:3,3]

    
    alpha = n1 @ n2 #if the vector are colinear alpha will be equal to one (since n1 and n2 are normalized), can happend if we consider two parralel slice
    beta =  n1 @ t1
    gamma = n2 @ t2

    
    if abs((1 - alpha*alpha))<1e-6: #if the vector are colinear, there is no intersection
        return 0,0,0
    a = 1/(1 - alpha*alpha)
    g = a*(beta - alpha*gamma)
    h = a*(gamma - alpha*beta)


   
    #line equation
    coeff = np.cross(n1,n2)
    pt = g*n1 + h*n2
   
    return coeff, pt, 1
 
  
def intersectionSegment(sliceimage,coeff,pt):
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
    M = sliceimage.get_transfo()
    
    Minv = np.linalg.inv(M)
    rinv = np.linalg.inv(M[0:3,0:3])
    n = rinv @ coeff
    pt = np.concatenate((pt,np.array([1])))
    ptimg =Minv @ pt
   
    a = -n[1]
    b = n[0]
    c = -(a*ptimg[0] + b*ptimg[1])
    
    
    #Intersection with the plan
    intersection = np.zeros((4,2)) #2 points of intersection of coordinates i,j,k 
    intersection[3,:] = np.ones((1,2))
    width = sliceimage.get_slice().shape[0]-1
    height = sliceimage.get_slice().shape[1]-1
 

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
        return 0,0
        
    #Compute the intersection point coordinates in the 3D space
    interw = np.zeros((4,2)) #2 points of intersection, with 3 coordinates x,y,z
    interw[3,:] = np.ones((1,2))
    interw = M @ intersection 
    
    interw[0:3,0] = interw[0:3,0] - pt[0:3]
    interw[0:3,1] = interw[0:3,1] - pt[0:3]
    
    squareNorm = coeff @ coeff.transpose()
    lambdaPropo = (((1/squareNorm) * coeff.transpose()) @ interw[0:3,:]) 

    
    return lambdaPropo,1


    
    
def minLambda(lambdaPropo1,lambdaPropo2,intersection='union'):
    
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
    if intersection=='union':
        lambdaMin[0] = min(min(lambdaPropo1),min(lambdaPropo2))
        lambdaMin[1] = max(max(lambdaPropo1),max(lambdaPropo2))

    
    elif intersection=='intersection':
        lambdaMin[0] = max(min(lambdaPropo1),min(lambdaPropo2))
        lambdaMin[1] = min(max(lambdaPropo1),max(lambdaPropo2))

    else:
        print(intersection, ' is not recognize : choose either intersection or union')
    
    return lambdaMin #Return 2 values of lambda that represent the common segment between the 2 slices

def commonSegment(Slice1,Slice2,intersection='union'):
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
    
    M1 = Slice1.get_transfo()
    M2 = Slice2.get_transfo()
    
    coeff,pt,ok = intersectionLineBtw2Planes(M1,M2)

    
    if ok<1: #if there is no intersection lines (the 2 planes are parralel) it is useless to compute the intersectionSegment
        return 0,0,0,0
    
    
    lambdaPropo1,ok = intersectionSegment(Slice1,coeff,pt) #if there is no intersection segment (the line of intersection is outisde of the image or on a corner), it useless to compute a common segment

   
    if ok<1:
        return 0,0,0,0
    
    lambdaPropo2,ok = intersectionSegment(Slice2,coeff,pt)

    
    if ok<1:
        return 0,0,0,0
    

    lambdaMin = minLambda(lambdaPropo1,lambdaPropo2,intersection)

        
    if lambdaMin[0]==lambdaMin[1]: #the segment is nul, there is no intersection
        return 0,0,0,0
        
    M1 = Slice1.get_transfo()

    M2 = Slice2.get_transfo()

    point3D = np.zeros((3,2))
    
    
    point3D[0:3,0] = lambdaMin[0] * coeff + pt #Point corresponding to the value of lambda
    point3D[0:3,1] = lambdaMin[1] * coeff + pt
    
    point3D = np.concatenate((point3D,np.array([[1,1]])))
    
        
    pointImg1 = np.zeros((4,2))
    pointImg1[3,:] = np.ones((1,2))
    
    
    pointImg2 = np.zeros((4,2))
    pointImg2[3,:] = np.ones((1,2))
    
    
    pointImg1 = np.linalg.inv(M1) @ point3D

    pointImg2 = np.linalg.inv(M2) @ point3D 
    
    distance1 = np.linalg.norm(pointImg1[0:2,0] - pointImg1[0:2,1]) #distance between two points on the two images
    distance2 = np.linalg.norm(pointImg2[0:2,0] - pointImg2[0:2,1]) 

    res = 1
    #min(Slice1.get_slice().header.get_zooms()) #the smaller resolution of a voxel
        
    if res<0: #probmem with the resolution of the image
        return 0,0,0,0
        
    if max(distance1,distance2)<1: #no pixel in commun
        return 0,0,0,0
        
    nbpoint = int(np.round(max(distance1,distance2)+1)/res) #choose the max distance and divide it by the smaller resolution 
        
    return pointImg1[0:3],pointImg2[0:3],nbpoint,1


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
        return 0,0
    interpol= np.zeros(nbpoint)
    interpolMask = np.zeros(nbpoint)
    pointInterpol = np.zeros((2,nbpoint))
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
        return 0,0
    
    valindex=np.linspace(0,nbpoint-1,nbpoint,dtype=int)
    if intersection=='union':
        index = index1+index2 
    
    elif intersection=='intersection':
        index = index1*index2
    
    else:
        print(intersection, "is not recognized, choose either 'itersection' or 'union'")
    
    index = valindex[index==True]
    commonVal1 = np.zeros(val1.shape[0])
    commonVal1[index1] = val1[index1]
    commonVal2 = np.zeros(val2.shape[0])
    commonVal2[index2] = val2[index2]
 
    

    return commonVal1[index],commonVal2[index],index
    

def loadSlice(img,mask,listSlice,orientation):
    """
    Create Z sliceObject from the slice and the mask, with z being the third dimension of the 3D image
    """
    mask = mask.get_fdata().squeeze()
    normalizedImg = normalization(img)
    img = normalizedImg
    for z in range(img.shape[2]): #Lecture des images
    
        slice_img = img.get_fdata()[:,:,z]
        slice_mask = mask[:,:,z]
        if ~(np.all(slice_mask==0)):
            mz = np.eye(4)
            mz[2,3]= z
            sliceaffine = img.affine @ mz
            nifti = nib.Nifti1Image(slice_img,sliceaffine)
            c_im1 = SliceObject(nifti,slice_mask,orientation)
            listSlice.append(c_im1)
            
        
    return listSlice


def error(commonVal1,commonVal2):
        return sum((commonVal1 - commonVal2)**2)
    
def totalError(previousError,newError,previousNbpoint,newnbpoint):
        var_error = previousError + newError
        var_nbpoint = previousNbpoint + newnbpoint
        return var_error,var_nbpoint

    
def f_mse(var_error,var_nbpoint):
    """
    FUnction to compute the MSE

    """
    mse = var_error/var_nbpoint
    return mse


def indexMse(slice1,listSlice):
    """
    The function computes the Mean Square Error between slice 1 and its orthogonal slices
    and return an array of the MSE between slice1 and each slice

    Inputs :
    slice1 : 
        type slice, contains all the necessary information about the slice
    listSlice : 
        list of type slice, contains the images in the three orientations, axial,sagital and coronal

    Ouptputs : 
    DICE :
        Array containing the MSE between slice1 and its orthogonal slices
    NBPOINT : 
        Array contaning the number of point in the union between each point

    """
    MSE = np.zeros(len(listSlice))
    NBPOINT = np.zeros(len(listSlice))
    indice = 0
    for slice2 in listSlice:
        if slice1.get_orientation() != slice2.get_orientation(): #there is no intersection between slices and its orthogonal slices
            newError,commonPoint = MSElocal(slice1,slice2)
            MSEloc= newError
            MSE[indice]=MSEloc
            NBPOINT[indice] = commonPoint
            indice = indice+1
            
            # pointImg1,pointImg2,nbpoint,ok = commonSegment(slice1,slice2)
            # if ok>0:
            #     val1,index1,nbpointSlice1=sliceProfil(slice1, pointImg1, nbpoint)  
            #     val2,index2,nbpointSlice2=sliceProfil(slice2, pointImg2, nbpoint)
            #     commonVal1,commonVal2,index = commonProfil(val1, index1, val2, index2,nbpoint)
            #     commonPoint = index.shape[0]
            #     if index.size != 0 and ~(index1.any()==False) and ~(index2.any()==False):
            #         newError = error(commonVal1,commonVal2)
            #         MSE[indice] = newError/commonPoint
            #     NBPOINT[indice] = commonPoint
            #     indice = indice+1
            
    return MSE[0:indice],NBPOINT[0:indice] 

# def indexGlobalMse(listSlice):
#     """
#     The function computes the means square error for each slice between a slice and its orthogonal slices
#     and return them in an array

#     Inputs :
#     listSlice : list of slice that represent our 3D volume
        
#     Returns :
#     MSE_GLOB : array of double
#         array, each values represent de MSE between one slice and its orthogonal slices
#       NBPOINT_GLOB : array of integers
#         array, each values represent the total number of intersection point between one slice and its orthogonal slices

#     """
#     MSE_GLOB = np.zeros(len(listSlice))
#     NBPOINT_GLOB= np.zeros(len(listSlice))
#     indice=0
#     for slice1 in listSlice:
#         mse,nbpoint_mse = (slice1,index, listSlice)
#         NBPOINT_GLOB[indice] = nbpoint_mse
#         MSE_GLOB[indice] = mse
#         indice=indice+1
#     return MSE_GLOB,NBPOINT_GLOB       

def indexGlobalMse(listSlice):
    
     MSE_GLOB = np.zeros(len(listSlice))
     NBPOINT_GLOB= np.zeros(len(listSlice))
     indice=0
     for slice1 in listSlice:
         errorSlice, nbpointSlice = indexMse(slice1,listSlice)
         globalErrorSlice = sum(errorSlice)
         globalErrornbpointSlice = sum(nbpointSlice)
         NBPOINT_GLOB[indice] = globalErrorSlice
         MSE_GLOB[indice] = globalErrornbpointSlice
         indice=indice+1
    # size = gridError.shape
    # nbslice = size[0]
    # rowError = gridError[indexSlice,0:indexSlice]
    # rowNbpoint = gridNbpoint[indexSlice,0:indexSlice]
    # columnError = gridError[indexSlice:nbslice-1,indexSlice]
    # columnNbpoint = gridNbpoint[indexSlice:nbslice-1,indexSlice]
    # globalError = np.concatenate((rowError,columnError.T))
    # globalNbpoint = np.concatenate((rowNbpoint,columnNbpoint.T))
    
     return NBPOINT_GLOB[0:indice],MSE_GLOB[0:indice]

def MSElocal(slice1,slice2):
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
    pointImg1,pointImg2,nbpoint,ok = commonSegment(slice1,slice2)
    newError=0
    commonPoint=0
    if ok>0:
        val1,index1,nbpointSlice1=sliceProfil(slice1, pointImg1, nbpoint)  
        val2,index2,nbpointSlice2=sliceProfil(slice2, pointImg2, nbpoint)
        commonVal1,commonVal2,index = commonProfil(val1, index1, val2, index2,nbpoint)
        commonPoint = index.shape[0]
        if index.size != 0 and ~(index1.any()==False) and ~(index2.any()==False):
            newError = error(commonVal1,commonVal2)
    return newError,commonPoint


def updateCostBetweenAllImageAndOne(slice1,indexSlice,listSlice,gridCriterium,gridnbpoint):
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
    i_slice2 = 0
    #testError = 0
    
    for slice2 in listSlice:
        #if (indexSlice > i_slice2):
             if slice1.get_orientation() != slice2.get_orientation(): #there is no intersection between slices and its orthogonal slices
                    
                    newError,commonPoint = MSElocal(slice1,slice2)
                    if indexSlice > i_slice2 : 
                        gridCriterium[indexSlice,i_slice2] = newError
                        gridnbpoint[indexSlice,i_slice2] = commonPoint
                    else:
                        gridCriterium[i_slice2,indexSlice] = newError
                        gridnbpoint[i_slice2,indexSlice] = commonPoint
                    #testError = testError + newError
             i_slice2=i_slice2+1
    
    #debug procedure :
    nbSlice = len(listSlice)
    debugGridError,debugGridNbpoint = computeCostBetweenAll2Dimages(listSlice)
   
    if costFromMatrix(debugGridError,debugGridNbpoint) != costFromMatrix(gridCriterium,gridnbpoint):
        print("Rate!!!")
        return 0,0
    
    return gridCriterium,gridnbpoint #,testError

def MSEhisto(MSE):
    histoMse =  plt.hist(MSE,range=(min(MSE),max(MSE)),bins='auto')
    return histoMse

            
def dice(intersection,union):
    """
    Function to compute the DICE
    """
    dice = intersection/union
    return dice
 
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
    DICE = np.zeros(len(listSlice))
    indice= 0
    for slice2 in listSlice:
        if slice1.get_orientation() != slice2.get_orientation(): 
            #compute the intersection
            DICE[indice] = DICElocal(slice1,slice2)

   
    return DICE[0:indice]

def DICElocal(slice1,slice2):
    """
    The function computes a DICE (IoU) 2 slices

    Inputs:
    slice1 : 
        type slice, contains all the information about the slice
    slice2 : 
        type slice, contains all the information about the slice

    Returns
    intersection : integer
        number of point in the intersection
    union : integer
        number of point in the union
    DICE : double
        intersection over union

    """
    DICE = 0
    intersection=0
    union=0
    union_pt1,union_pt2,union_nbpoint,union_ok = commonSegment(slice1,slice2)
    if union_ok > 0:
        union_val1,union_index1,union_nbpointSlice1=sliceProfil(slice1, union_pt1, union_nbpoint)
        union_val2,union_index2,union_nbpointSlice2=sliceProfil(slice2, union_pt2, union_nbpoint)
        union_commonVal1,union_commonVal2,union_index = commonProfil(union_val1, union_index1, union_val2, union_index2, union_nbpoint)
        union = union_index.shape[0]
            
        #compute the union
        intersection_pt1,intersection_pt2,intersection_nbpoint,intersection_ok = commonSegment(slice1,slice2,"intersection")
        if intersection_ok > 0:
            intersection_val1,intersection_index1,intersection_nbpointSlice1=sliceProfil(slice1, intersection_pt1, intersection_nbpoint)
            intersection_val2,intersection_index2,intersection_nbpointSlice2=sliceProfil(slice2, intersection_pt2, intersection_nbpoint)
            intersection_commonVal1,intersection_commonVal2,intersection_index = commonProfil(intersection_val1, intersection_index1, intersection_val2, intersection_index2,intersection_nbpoint,"intersection")
            intersection = intersection_index.shape[0]
        if intersection != 0 and union !=0:
            DICE = dice(intersection,union)
    return intersection,union,DICE

def DICEGlobal(slice1,listSlice): #union of intersection under union of union
    """
    Computes a DICEglobal,the dice is the union of interesction over the union of union. The function computes the dice between slice1 and its orthogonal slices
    """

    globalInter = 0
    globalUnion = 0
    globalDice = 0
    for slice2 in listSlice:
        inter, union, DICE = DICElocal(slice1,slice2)
        globalInter = globalInter + inter
        globalUnion = globalUnion + union
    globalDice =    dice(globalInter,globalUnion)
    return globalInter,globalUnion,globalDice
    
# def globalCriteriumDice(listSlice):
#     """
#     Computes the dice between each slices and its orthogonal slices

#     Inputs
#     listSlice : list of slices which represents the 3D volume
        

#     Returns
#     unionInter : 
#         nb of element in the intersection of each slices
#     unionUnion : TYPE
#         nb of element in the union of each slices


#     """
    
#     nbSlice = len(listSlice)
#     L2 = np.zeros(nbSlice,nbSlice)
#     nbpoint_inter = np.zeros(nbSlice,nbSlice)
#     i_slice1=0
    
#     unionInter = 0
#     unionUnion = 0
    
#     for slice1 in listSlice:
#         i_slice2=0
#         for slice2 in listSlice:
#             if slice1.get_orientation() != slice2.get_orientation():
#                 if already_done[i_slice1,i_slice2] == already_done[i_slice2,i_slice1] and already_done[i_slice2,i_slice1] <= 0 :
#                     Inter,Union,DICEloc = DICElocal(slice1,slice2)
#                     unionInter = unionInter + Inter
#                     unionUnion = unionUnion + Union
#                     already_done[i_slice1,i_slice2] = 1
#                     already_done[i_slice2,i_slice1] = 1   
                     
#             i_slice2 = i_slice2+1
#         i_slice1 = i_slice1+1
#     if  unionUnion == 0:
#         return unionInter,unionUnion,0
#     DICE = dice(unionInter,unionUnion)
#     return unionInter,unionUnion,DICE
    
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
    gridError = np.zeros((nbSlice,nbSlice))
    gridNbpoint = np.zeros((nbSlice,nbSlice))
    i_slice1=0
    
    for slice1 in listSlice:
        i_slice2=0
        for slice2 in listSlice:
            if (i_slice1 > i_slice2):
                if slice1.get_orientation() != slice2.get_orientation():
                        newError,commonPoint = MSElocal(slice1,slice2)
                        gridError[i_slice1,i_slice2] = newError
                        gridNbpoint[i_slice1,i_slice2] = commonPoint
                    
            i_slice2 = i_slice2+1
        i_slice1 = i_slice1+1
    
    return gridError,gridNbpoint     

def cost_fct(x, slice, indexSlice, listSlice, gridError, gridNbpoint): #slice and listSlice are constant parameters and x is variable
    """
    Compute the cost function. 
    
    Inputs :
        x : 6D array
            parameters of the rigid transformation. The first three parameters represent the rotation and the last three parameters represent the translation
    """
    print('parameters before : ',slice.get_parameters())
    slice.set_parameter(x)
    print('parameters after : ',slice.get_parameters())
    gridErrorUpdated, gridNbpointUpdated = updateCostBetweenAllImageAndOne(slice, indexSlice, listSlice, gridError, gridNbpoint)
    res = costFromMatrix(gridErrorUpdated,gridNbpointUpdated)
    return res

def costFromMatrix(gridError,gridNbpoint):
    globalError = np.sum(gridError)
    globalNbpoint = np.sum(gridNbpoint)
    MSE = globalError/globalNbpoint
    return MSE
#def cost_fct()


def cost_fct_dice(x,slice,listSlice,unionInter,unionUnion):
    
    previous_inter,previous_union,previous_dice = DICEGlobal(slice, listSlice)
    slice.set_parameter(x)
    new_inter,new_union,new_dice = DICEGlobal(slice, listSlice)
    res_inter = unionInter-previous_inter+new_inter
    res_union = unionUnion-previous_union+new_union
    res = res_inter/res_union
    return res
    
def normalization(image):
    """
    Normalized the images to a standard normal distribution

    Inputs
    image : 3D matrix
        The image we want to normalized

    Returns
    normalizedImg : 3D matrix
        The normalized image

    """
    mean = np.mean(image.get_fdata())
    std =  np.std(image.get_fdata())
    newImage = (image.get_fdata() - mean)/std
    normalizedImg = nib.Nifti1Image(newImage,image.affine)
    return normalizedImg
                    
#     #map_coordinates(input, coordinates)
    
# def erreur(commonVal1,commonVal2):
    
#     erreur = 0
    
#     for i in range(commonVal1.shape[0]):
#         e = abs(commonVal1[i] - commonVal2[i])
#         erreur = erreur + e 
        
#     return erreur, commonVal1.shape[0], erreur/commonVal1.shape[0]

# def optimization(erreur,N,Slice):
#     minimize(erreur/N,Slice.get_parameters,)

# #Fonction qui interpole les valeurs d intensite sur le segment d intersection et renvoit le profil d intensite
# def SliceProfil(Coupe,PtsegmentCoupe,nbpoint):
    
    
#     #nbpointx = np.abs(int(PtsegmentCoupe[0,0]) - int(PtsegmentCoupe[0,1])) #nb de points du segment
#     x = np.linspace(int(PtsegmentCoupe[0,0]), int(PtsegmentCoupe[0,1]), num=nbpoint, endpoint=True, retstep=False, dtype=int, axis=0)  #donne les coordonnees des points de x du segments
#     y = np.linspace(int(PtsegmentCoupe[1,0]), int(PtsegmentCoupe[1,1]), num=nbpoint, endpoint=True, retstep=False, dtype=int, axis=0)  #donne les coordonnees des points de y du segments
#     masque = Coupe.get_masque()
#     #Coupe_sortie = Coupe.get_data().copy()
#     val = np.zeros((nbpoint,2))
    
#     i=0
#     for i in range(x.shape[0]):
#         if masque.get_fdata()[x[i],y[i]]>0:
#            val[i,:] = [x[i],y[i]]
#             #Coupe_sortie[x[i],y[i]] = 0
    
#     #nifti_sortie = nib.Nifti1Image(Coupe_sortie,Coupe1.get_affine())
    
#     print(np.transpose(val))
#     print(x)
#     print(y)
#     sortie = np.zeros(nbpoint)
#     #masque_sortie = np.zeros(nbpoint)
    
#     map_coordinates(Coupe.get_data(),np.transpose(val),output=sortie, order=1, mode='constant', cval=0.0, prefilter=True)
    
#     index = sortie>0
    
#     return sortie,index


# # def CalculErreur(Profil1,Profil2,nbpoint):
        
# #     erreur = 0
        
# #     for i in range(nbpoint-1):
# #         erreur = erreur + (Profil1[i]-Profil2[i])*(Profil1[i]-Profil2[i])
        
# #     return erreur 
    





# ###Test : 
    
Coupe_img = []
Coupe_img1 = loadSlice(img1, masque1, Coupe_img,'axial')    
Coupe_img2 = loadSlice(img2, masque2, Coupe_img,'coronal')    
Coupe_img3 = loadSlice(img3, masque3, Coupe_img,'sagital')    

# Coupe1 = Coupe_img1[7]

# Coupe2 = Coupe_img2[19]



# coupe1 = Coupe_img1[9]
# previousError = 0
# previousNbpoint = 0
# MSE = np.zeros(len(Coupe_img2))
# DICE = np.zeros(len(Coupe_img2))
# NBPOINT = np.zeros(len(Coupe_img2))
# indice = 0
# for coupe2 in Coupe_img2:
#         newerror=0
#         dicetmp=0
#         pointImg1,pointImg2,nbpoint,ok = commonSegment(coupe1,coupe2)
#         interPt1,interPt2,interNbpoint,ok = commonSegment(coupe1,coupe2,"intersection")
#         if ok>0:
#             val1,index1,nbpointSlice1=sliceProfil(coupe1, pointImg1, nbpoint)
#             interVal1,interIndex1,interNbpointSlice1=sliceProfil(coupe1, interPt1, interNbpoint)
#             val2,index2,nbpointSlice2=sliceProfil(coupe2, pointImg2, nbpoint)
#             interVal2,interIndex2,interNbpointSlice2=sliceProfil(coupe2, pointImg2, nbpoint)
#             commonVal1,commonVal2,index = commonProfil(val1, index1, val2, index2,nbpoint)
#             interCommonVal1,interCommonVal2,interIndex = commonProfil(val1, index1, val2, index2,nbpoint,"intersection")
#             commonPoint = index.shape[0]
#             interCommonPoint = interIndex.shape[0]
#             #Before applying the mask
#             title1 = 'Intersection segment for image1, without mask'
#             title2 = 'Intersection segment for image2, without mask'
#             plotsegment(coupe1,pointImg1,ok,nbpoint,title1,mask=np.nan,index=np.nan,nbpointSlice=None)
#             plotsegment(coupe2,pointImg2,ok,nbpoint,title2,mask=np.nan,index=np.nan,nbpointSlice=None)
   
#             #After applying the mask
#             title1 = 'Intersection segment for image1, with mask'
#             title2 = 'Intersection segment for image2, with mask'
#             plotsegment(coupe1,pointImg1,ok,nbpoint,title1,mask=coupe1.get_mask(),index=index,nbpointSlice=nbpointSlice1)
#             plotsegment(coupe2,pointImg2,ok,nbpoint,title2,mask=coupe2.get_mask(),index=index,nbpointSlice=nbpointSlice2)
            
#             if index.size != 0 and ~(index1.any()==False) and ~(index2.any()==False):
#                 displayIntensityProfil(commonVal1,index1,commonVal2,index2,index)
#                 newError = error(commonVal1,commonVal2)
#                 dicetmp = interCommonPoint/commonPoint
#                 print(interCommonPoint,commonPoint)
#             else:
#                 newError = 100000000
#                 commonPoint = nbpoint
#                 dicetmp = 0
#             previousError,previousNbpoint = totalError(previousError, newError, previousNbpoint, commonPoint)
#             MSE[indice] = newError/commonPoint
#             DICE[indice] = dicetmp
#             NBPOINT[indice] = commonPoint*(10**3)
#             indice = indice+1

# plt.figure()
# plt.plot(MSE)
# plt.plot(NBPOINT)
# plt.title('MSE')

# plt.figure()
# plt.plot(DICE)
# plt.title('DICE')

