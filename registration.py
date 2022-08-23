# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 13:59:55 2021

@author: mercier
"""


import nibabel as nib 
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize
import time
from tools import createVolumesFromAlist, computeMeanRotation, computeMeanTranslation, ParametersFromRigidMatrix, rigidMatrix, denoising, matrixOfWeight
from numba import jit
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
    
    lambdaMin[0] = max(min(lambdaPropo1),min(lambdaPropo2))
    lambdaMin[1] = min(max(lambdaPropo1),max(lambdaPropo2))
   

    
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
    
    index =~np.isnan(interpol) * interpolMask>0
    #val_mask = interpol * interpolMask
    #index=val_mask>0
      
    return interpol,index,np.shape(index[index==True])[0]


def commonProfil(val1,index1,val2,index2,nbpoint,intersection='union'):
    """
    
    Compute the intensity of points along the intersection in two orthogonal slice.
    
    Inputs :
    
    val1 : nbpointx1 vector
        values of intensity in the first slice
    index1 : nbpointx1 vector
        values of interest in the first slice
    val2 : nbpointx1 vector
        values of intensity in the seconde slice
    index2 : nbpointx1 vector
        values of interest in the second slice
    intersection : string
        If intersection is union, it return the union of the intersection. Values of the points outside the mask are the same values as in the image. Values outside the image are set to zeros.
        By default, the intersection is set to union
        
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


    commonVal1 = np.zeros(val1.shape[0])  
    commonVal1[~np.isnan(val1)] = val1[~np.isnan(val1)]
    commonVal2 = np.zeros(val2.shape[0])  
    commonVal2[~np.isnan(val2)] = val2[~np.isnan(val2)] 
 
    return commonVal1[index],commonVal2[index],index
    

@jit(nopython=True)
def error(commonVal1,commonVal2):
    """
    Compute sumed squares error between two intensity profils. The function is accelerated with numba.

    Inputs : 
        
    commonVal1 : 1-size vector
        Intensity profile in slice 1
    commonVal2 : 1-size vector
        Intensity profile in slice Z

    Outputs : 
        returns the sumed squared error between the two profiles
        
    """
    return np.sum((commonVal1 - commonVal2)**2)
    



def costLocal(slice1,slice2):
    """
    
    Compute the MSE between two slices

    Inputs : 

    slice1 : a slice of a 3D volume
        type sliceObject, contains all the necessary information about the slice
    slice2 : slice
        type sliceObject, contains all the necessary information about the slice

    Returns :
        
    newError : double
        square error between two slices
    commonPoint : integer
        number of common point on the intersection between the two slices
    inter : integer
        number of points in the intersection of intersections
    union : integer
        number of pointsin the union of intersections

    """
    sliceimage1=slice1.get_slice().get_fdata();sliceimage2=slice2.get_slice().get_fdata();res=min(slice1.get_slice().header.get_zooms())
    M1=slice1.get_transfo();M2=slice2.get_transfo()
    pointImg1,pointImg2,nbpoint,ok = commonSegment(sliceimage1,M1,sliceimage2,M2,res)
    ok=np.int(ok[0,0]); nbpoint=np.int(nbpoint[0,0]) #ok and nbpoints are 2-size vectors to allow using numba with this function
    newError=0; commonPoint=0; inter=0; union=0
    
    if ok>0:
        val1,index1,nbpointSlice1=sliceProfil(slice1, pointImg1, nbpoint)  #profile in slice 1
        val2,index2,nbpointSlice2=sliceProfil(slice2, pointImg2, nbpoint)  #profile in slice 2
        commonVal1,commonVal2,index=commonProfil(val1, index1, val2, index2,nbpoint) #union between profile in slice 1 and 2 (used in the calcul of MSE and DICE)
        val1inter,val2inter,interpoint=commonProfil(val1, index1, val2, index2,nbpoint,'intersection') #intersection between profile in slice 1 and 2 (used in the calcul of DICE)
        commonPoint=index.shape[0] #numbers of points along union profile (for MSE)
        newError=error(commonVal1,commonVal2) #sumed squared error between the two profile
        inter=interpoint.shape[0] #number of points along intersection profile (for DICE)
        union=index.shape[0] #number of points along union profile (for DICE)
    return newError,commonPoint,inter,union


def updateCostBetweenAllImageAndOne(indexSlice,listSlice,gridError,gridNbpoint,gridInter,gridUnion):
    """
    The function update the two matrix used to compute MSE, when one slice position is modified

    Inputs:
    slice1 : slice
        type slice, contains all the necessary information about the slice
    listSlice : list of slices
        contains the slices from all stacks

    """
    
    slice1=listSlice[indexSlice] #the slice that moves in the list


        
    for i_slice2 in range(len(listSlice)):
        
        slice2=listSlice[i_slice2] #one slice in the list
                
        if slice1.get_orientation() != slice2.get_orientation(): #we compute the error between slice1 and slice2 only if they are not from the same stacks

            if indexSlice > i_slice2 : #The error matrix is triangular as error between slice i and j is the same than between slice j and i. We choose to enter only the values when i>j.
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
    ##check that we obtain the same result when we only change one line in the error matrix and when we compute the entire matrix
    # debugGridError,debugGridNbpoint, debugGridInter, debugGridUnion = computeCostBetweenAll2Dimages(listSlice)


    # if costFromMatrix(debugGridError,debugGridNbpoint) != costFromMatrix(gridError,gridNbpoint):
    #     print('error')
    #     return 0
    # if costFromMatrix(debugGridInter,debugGridUnion) != costFromMatrix(gridInter,gridUnion):
    #     print('error')
    #     return 0





def computeCostBetweenAll2Dimages(listSlice):
    """
    Computes four matrix used in calcul of MSE and DICE. 
    

    Inputs:
    
    listSlice : list of slices, contains slices from all stacks
    
    Returns:
    
    gridError : Triangular matrix which contains summed squares error between slices, ex : gridError[1,0] contains the summed squared error between slices 1 and 0 in listSlice  
    gridNbpoint : Triangular matrix which contains  the number of points on the intersection between slices, ex : gridNbpoint[1,0] contains the number of points in the intersection between slices 1 and 0 in listSlice
    gridInter : Triangular matrix which contains the number of points on the intersection of intersection between slices
    gridUnion : Triangular matrix which contains the number of points on the union of intersection between slices
    
   
    """
    #Initialization
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
                        newError,commonPoint,inter,union=costLocal(slice1,slice2) #computed cost informations between two slices
                        gridError[i_slice1,i_slice2]=newError 
                        gridNbpoint[i_slice1,i_slice2]=commonPoint
                        gridInter[i_slice1,i_slice2]=inter
                        gridUnion[i_slice1,i_slice2]=union
                    
    return gridError,gridNbpoint,gridInter,gridUnion     





def cost_fct(x,i_slice, listSlice, gridError, gridNbpoint, gridInter, gridUnion, gridWeight,lamb): #slice and listSlice are constant parameters and x is variable
    """
    Compute the cost function, the one we want to optimize in function of x.
    
    Inputs :
        x : 6D array
            parameters of the rigid transformation. The first three parameters represent the rotation and the last three parameters represent the translation
    """
    
    slicei=listSlice[i_slice]
    slicei.set_parameters(x) #update parameters in the slice of interest
    updateCostBetweenAllImageAndOne(i_slice, listSlice, gridError, gridNbpoint, gridInter, gridUnion) #update matrix used to compute MSE and DICE
    mse = costFromMatrix(gridError,gridNbpoint) #value of MSE compute on all slices
    dice = costFromMatrix(gridInter, gridUnion) #value of DICE computed on all slices
    
    return mse  - lamb*(dice)



def costFromMatrix(gridNum,gridDenum):
    
    """
    Compute the cost on all slices from two matrix. Can be equally used to compute the MSE and the DICE
    
    """
    
    
    globalNum = np.sum(gridNum)
    globalDenum= np.sum(gridDenum)
    if globalDenum>0:
        cost = globalNum/globalDenum
    else:
        cost=0
    return cost



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
                if data[x,y,0]> 1e-10:
                    mean_sum = mean_sum + data[x,y,0]
                    n = n + 1
                    var.append(data[x,y,0])
    mean = mean_sum/n
    std = np.sqrt((sum((np.array(var)-mean)*(np.array(var)-mean))/(n)))
    
    listSliceNorm = []
    for s in listSlice:
        slices = s.get_slice().get_fdata() 
        X,Y,Z = slices.shape
        newSlice = np.zeros((X,Y,1))
        for x in range(X):
            for y in range(Y):
                newSlice[x,y,0] = (slices[x,y,0] - mean)/std
        newNifti = nib.Nifti1Image(newSlice, s.get_slice().affine)
        s.set_slice(newNifti)
        listSliceNorm.append(s)

    #dbug : 
    #check if the mean and standart deviation are 0 and 1 after normalization
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
 

def global_optimisation(listSlice):
    """
    Compute the optimised parameters for each slice. At the end of the function parameters of each slice must be the optimised parameters. The function returns the evolution of the registration on each iterarion.
    
    Input
    
    listSlice : list of sliceObjects
        list of slices from all staks

    Returns
    
    dicRes : A dictionnay wich contains informations on the evolution of the registration during all iteration (ex : evolution of MSE, evolution of DICE, evolution of parameters ....)
    rejectedSlice : list of rejected slices and their corresponding stack
    
    """
    
    #Initialisation
    nbSlice=len(listSlice) 
    EvolutionError=[] 
    EvolutionDice=[]
    EvolutionGridError=[] 
    EvolutionGridNbpoint=[]
    EvolutionGridInter=[]
    EvolutionGridUnion=[]
    EvolutionParameters=[] 
    Previous_parameters=np.zeros((6,nbSlice))
    EvolutionTransfo=[]
    
   
    dicRes={}
    for i in range(nbSlice): 
        slicei=listSlice[i]
        EvolutionParameters.extend(slicei.get_parameters())
        Previous_parameters[:,i]=slicei.get_parameters()
        EvolutionTransfo.extend(slicei.get_transfo())
        
    dicRes["evolutionparameters"] = EvolutionParameters
    dicRes["evolutiontransfo"] = EvolutionTransfo
    gridError,gridNbpoint,gridInter,gridUnion=computeCostBetweenAll2Dimages(listSlice) 
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
    

    #optimised the cost for all slices
    gridError,gridNbpoint,gridInter,gridUnion,dicRes=SlicesOptimisation(listSlice,0.1, gridError, gridNbpoint, gridInter, gridUnion, dicRes)
    
    #corrected slices bad register
    dicRes=MeanOptimisation(listSlice,gridError,gridNbpoint,gridInter,gridUnion,dicRes)
    
    #reject slices bad register
    dicRes,rejectedSlices=LastOptimisation(listSlice,0.1,gridError,gridNbpoint,gridInter,gridUnion,dicRes)
    
    
    return dicRes, rejectedSlices
    

def SimplexOptimisation(delta,xatol,x0,i_slice,listSlice,gridError,gridNbpoint,gridInter,gridUnion,initial_s,lamb):
    """
    Implementation of the simplex (Nealder - Mead) method for the problem. We used the simplex method implemented in scipy
    
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
    x0 = slicei.get_parameters()
    #delta=5
    P0 = x0 #create the initial simplex
    P1 = P0 + np.array([delta,0,0,0,0,0])
    P2 = P0 + np.array([0,delta,0,0,0,0])
    P3 = P0 + np.array([0,0,delta,0,0,0])
    P4 = P0 + np.array([0,0,0,delta,0,0])
    P5 = P0 + np.array([0,0,0,0,delta,0])
    P6 = P0 + np.array([0,0,0,0,0,delta])

                                    
    initial_s[0,:]=P0
    initial_s[1,:]=P1
    initial_s[2,:]=P2
    initial_s[3,:]=P3
    initial_s[4,:]=P4
    initial_s[5,:]=P5
    initial_s[6,:]=P6

         
    nbSlice=len(listSlice)                                       
    X,Y = gridError.shape
    NM = minimize(cost_fct,x0,args=(i_slice,listSlice,gridError,gridNbpoint,gridInter,gridUnion,np.ones((nbSlice,nbSlice)),lamb),method='Nelder-Mead',options={"disp" : False, "maxiter" : 2000, "maxfev":1e4, "xatol" : 1e-1, "fatol" : 1e-2, "initial_simplex" : initial_s , "adaptive" :  True})
    #optimisation of the cost function using the simplex method                                    
    
    x_opt = NM.x #best parameter obtains after registration
    slicei.set_parameters(x_opt)
    updateCostBetweenAllImageAndOne(i_slice, listSlice, gridError, gridNbpoint,gridInter,gridUnion) #updated matrix of cost
    costMse=costFromMatrix(gridError,gridNbpoint) #MSE after optimisation
    costDice=costFromMatrix(gridInter,gridUnion) #Dice after optimisation
    
    return costMse,costDice,gridError,gridNbpoint,gridInter,gridUnion,x_opt
    
def OptimisationThreshold(gridError,gridNbpoint):
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
    
    vectMse = vectMse[~np.isnan(vectMse)]                    
    valMse = np.median(vectMse)
    
    std = 1.4826*np.median(np.abs(vectMse-valMse)) #the distribution of MSE is approximately gaussien so we take the 95% intervall
    threshold = np.mean(vectMse) + 2*std
    
    return threshold


def updateResults(dicRes,gridError,gridNbpoint,gridInter,gridUnion,costMse,costDice,listSlice,nbSlice):
    
    """
    The function update the results at each iteration of the optimisation function
    """
    
    
    dicRes["evolutionerror"].append(costMse) #evolution of the MSE
    dicRes["evolutiondice"].append(costDice) #evolution of the DICE
    dicRes["evolutiongriderror"].extend(gridError.copy()) #evolution of the matrix error (summed squared error) (to compute the mse)
    dicRes["evolutiongridnbpoint"].extend(gridNbpoint.copy()) #evolution of the matrix of intersection point (to compute the mse)
    dicRes["evolutiongridinter"].extend(gridInter.copy()) #evolution of the matrix of intersection of intersection points (to compute the dice)
    dicRes["evolutiongridunion"].extend(gridUnion.copy()) #evolution of the matrix of union of intersection points (to compute the dice)
    
                      
    for i in range(nbSlice):
        slicei=listSlice[i]
        dicRes["evolutionparameters"].extend(slicei.get_parameters()) #evolution of the parameters
        dicRes["evolutiontransfo"].extend(slicei.get_transfo()) #evolution of the global matrix applied to the image

        
def SlicesOptimisation(listSlice,xatol,gridError,gridNbpoint,gridInter,gridUnion,dicRes):
    """
    
    Optimise the cost function for each slices

    Input:
        
    listSlice : list of sliceObjects, list of the 2D images of the fetus brains with the orientation information associated with each image
    gridError : triangular matrix representing the square error between each slices
    gridNbpoint : triangular matrix representing the number of common point between each slices
    gridInter : triangular matrix representing the number of point on the intersection between each slices
    gridUnion : triangular matrix representing the number of point on the union between each slices
    

    Ouputs:
        
    gridError : triangular matrix representing the square error between each slices
    gridNbpoint : triangular matrix representing the number of common point between each slices
    gridInter : triangular matrix representing the number of point on the intersection between each slices
    gridUnion : triangular matrix representing the number of point on the union between each slices
    dicRes : contains the updapted information about the evolution of registration

    """
    
    #maximum size of simplex
    #Initialisation
    initial_s = np.zeros((7,6)) 
    nbSlice = len(listSlice)
    images,mask=createVolumesFromAlist(listSlice)
    nbImages=len(images)
    nbSlice=len(listSlice)
    maxIt=0
    maxItSlice=10

    listSliceOr = listSlice.copy()
    while maxItSlice>1 and maxIt<10:
        
        #A gaussien filter on first iteration help to not fell into local minima
        if maxIt==0:
            for i_slice in range(nbSlice) :
                listSliceOr[i_slice].set_parameters(listSlice[i_slice].get_parameters())
            blurlist = listSliceOr.copy()
            listSlice = denoising(blurlist,3.0)
            gridError,gridNbpoint,gridInter,gridUnion = computeCostBetweenAll2Dimages(listSlice)
            lamb=10
            d=5
        
        #We reduce the value of blur of gaussien filter in next iteration
        elif maxIt==1:
            for i_slice in range(nbSlice) :
                 listSliceOr[i_slice].set_parameters(listSlice[i_slice].get_parameters())               
            listSlice = denoising(listSliceOr.copy(),1.0)
            gridError,gridNbpoint,gridInter,gridUnion = computeCostBetweenAll2Dimages(listSlice)
            costDice = costFromMatrix(gridInter,gridUnion)
            lamb=10
            d=5
        
        
        #Optimisation is done on the original image (no blurring)
        elif maxIt==2:
            for i_slice in range(nbSlice) :
                listSliceOr[i_slice].set_parameters(listSlice[i_slice].get_parameters())
            listSlice=listSliceOr.copy()
            gridError,gridNbpoint,gridInter,gridUnion = computeCostBetweenAll2Dimages(listSlice)
            lamb=10
            d=5
            
        
        #The dice is not taken in next iteration
        else :
            gridError,gridNbpoint,gridInter,gridUnion = computeCostBetweenAll2Dimages(listSlice)
            lamb=0
            d=5
            
        finish=np.zeros(nbSlice)
        maxItSlice=0
                        
        while finish.all()!=1 and maxItSlice<10: #we iterate until slices stop moving, in case they never stop, we stop the algorithm after 10 iteration
                    
            index_pre=0
            for n_image in range(nbImages):
                start = time.time()
                nbSliceImageN=len(images[n_image])
                randomIndex= np.arange(nbSliceImageN) 
                index=randomIndex+index_pre
                        
                with Pool() as p: #all slices from the same stacks are register together (in the limit of the number of CPU on your machine)
                    tmpfun=partial(SliceOptimisation1Image,d,xatol,listSlice,gridError,gridNbpoint,gridInter,gridUnion,initial_s,finish,lamb) 
                    res=p.map(tmpfun,index)
                                            
                for i_slice in range(nbSliceImageN): #update parameters once the registration is done
                    listSlice[randomIndex[i_slice]+index_pre].set_parameters(res[i_slice][0])
                    finish[randomIndex[i_slice]+index_pre]=res[i_slice][1]
                gridError,gridNbpoint,gridInter,gridUnion=computeCostBetweenAll2Dimages(listSlice)
                costMse=costFromMatrix(gridError,gridNbpoint)
                costDice=costFromMatrix(gridInter,gridUnion)
                index_pre=index_pre+len(images[n_image])
                                            
            maxItSlice=maxItSlice+1

                                        
        print('delta :',  d, 'iteration :', maxIt, 'final MSE :', costMse, 'final DICE :', costDice)
        maxIt=maxIt+1 
        end = time.time()
        elapsed = end - start
        
        for i_slice in range(nbSlice) :
            listSliceOr[i_slice].set_parameters(listSlice[i_slice].get_parameters())
            
        gridError,gridNbpoint,gridInter,gridUnion = computeCostBetweenAll2Dimages(listSliceOr)
        costMse=costFromMatrix(gridError, gridNbpoint)
        costDice=costFromMatrix(gridInter,gridUnion)
        updateResults(dicRes,gridError,gridNbpoint,gridInter,gridUnion,costMse,costDice,listSlice,nbSlice)
        print('MSE: ', costMse)
        print('Dice: ', costDice) 
        print(f'Temps d\'exécution : {elapsed}')
            
    return gridError,gridNbpoint,gridInter,gridUnion,dicRes
    

def SliceOptimisation1Image(d,xatol,listSlice,gridError,gridNbpoint,gridInter,gridUnion,initial_s,finish,lamb,nImage):    
    """
    FUnction used in allSliceOptimisation to allow multiprocessing

   Inputs : 

    d : size of the simplex
    xatol : toletance on x
    listSlice : list of slice that contains all slices from the different stacks.
    gridError : matrix, contains the quared error between each pair of slices
    gridNbpoint : matrix, contains the number of points in the intersection between each pair of slices
    gridInter : matrix, contains the number of points in the intersection of intersection between each pair of slices
    gridUnion : matrix, contains the number of point in the union of intersection between each pair of slices
    gridWeight : matrix, indicates which slices are well-registered and wich one are not
    initial_s : the initial simplex
    finish : boolean, indicates when a slice is completly register or not
    lamb : coefficient of the DICE
    nImage : index of the image

    Putputs : 
        
    x_opt : corrected parameters
    finish_val : boolean, indicate if the slice is already register or not

    """
    delta = d 
    slicei=listSlice[nImage]
    x0=slicei.get_parameters()
    previous_cost=oneSliceCost(gridError, gridNbpoint, nImage) - lamb*oneSliceCost(gridInter, gridUnion, nImage)
    finish_val=finish[nImage]
    if finish_val==0:
        costMse,costDice,gridError,gridNbpoint,gridInter,gridUnion,x_opt=SimplexOptimisation(delta,xatol, x0, nImage, listSlice, gridError, gridNbpoint, gridInter, gridUnion, initial_s, lamb)
        current_cost=oneSliceCost(gridError, gridNbpoint, nImage) - lamb*oneSliceCost(gridInter, gridUnion, nImage)
        print('nbSlice :',nImage, 'costMse :', costMse, 'costDice :', costDice,'golbal cost ', costMse - lamb*costDice, 'previous_cost :', previous_cost, 'current_cost :', current_cost)

        if previous_cost<current_cost+1e-1:
            finish_val=1
    else:
        x_opt=x0

    return x_opt,finish_val          

def oneSliceCost(gridNum,gridDenum,i_slice):
    """
    Compute the cost of one slice, using the two cost matrix. Can be used equally for MSE and DICE
    """
    num=sum(gridNum[i_slice,:])+sum(gridNum[:,i_slice])
    denum=sum(gridDenum[i_slice,:])+sum(gridDenum[:,i_slice])
    if denum==0:
        cost=0
    else:
        cost=num/denum
    if np.isinf(cost):
        print('ALERT ERROR')
    return cost


   
def LastOptimisation(listSlice,xatol,gridError,gridNbpoint,gridInter,gridUnion,dicRes):
    """
    
    Do one last optimisation

    Input:
        
    listSlice : list of sliceObjects, list of the 2D images of the fetus brains with the orientation information associated with each image
    threshold : scalar, if the MSE of a slice is above this value, it is well-register, if not, it is badly-register
    gridError : triangular matrix representing the square error between each slices
    gridNbpoint : triangular matrix representing the number of common point between each slices
    gridInter : triangular matrix representing the number of point on the intersection between each slices
    gridUnion : triangular matrix representing the number of point on the union between each slices
    dicRes : Dictionnary which contains all the results from the registration

    Ouputs:
        
    gridError : triangular matrix representing the square error between each slices
    gridNbpoint : triangular matrix representing the number of common point between each slices
    gridInter : triangular matrix representing the number of point on the intersection between each slices
    gridUnion : triangular matrix representing the number of point on the union between each slices
    dicRes : Same dicRes than the one in input but with updated results

    """
    
    
    #initialisation
    d=5
    initial_s = np.zeros((7,6))
    nbSlice=len(listSlice)
    images,mask=createVolumesFromAlist(listSlice)
    nbImages=len(images)    
    finish=np.zeros(nbSlice)

    index_pre=0
    #do a last optimisation
    for n_image in range(nbImages):
        nbSliceImageN=len(images[n_image])
        randomIndex= np.arange(nbSliceImageN)
        index=randomIndex+index_pre
        with Pool() as p:
            tmpfun=partial(SliceOptimisation1Image,d,xatol,listSlice,gridError,gridNbpoint,gridInter,gridUnion,initial_s,finish,0) 
            res=p.map(tmpfun,index)
        for i_slice in range(nbSliceImageN):
                listSlice[randomIndex[i_slice]+index_pre].set_parameters(res[i_slice][0])
                finish[randomIndex[i_slice]+index_pre]=res[i_slice][1]
        gridError,gridNbpoint,gridInter,gridUnion=computeCostBetweenAll2Dimages(listSlice)
        costMse=costFromMatrix(gridError,gridNbpoint)
        costDice=costFromMatrix(gridInter,gridUnion)
        index_pre=index_pre+len(images[n_image])
   
    rejectedSlices = removeBadSlice(listSlice, gridInter, gridUnion)
    updateResults(dicRes,gridError,gridNbpoint,gridInter,gridUnion,costMse,costDice,listSlice,nbSlice)
                        
            
    return dicRes, rejectedSlices

def MeanOptimisation(listSlice,gridError,gridNbpoint,gridInter,gridUnion,dicRes):
    """
    Optimise the slices bad-registered by doing the mean of paramters of closer slices from the same stack
    
    """
    
    nbSlice=len(listSlice)
    threshold_pre=24
    threshold=OptimisationThreshold(gridError,gridNbpoint)
    threshold_dice=0.8
    
    while threshold_pre>(threshold + 0.01):
        print('threshold :',threshold)
        gridWeight=matrixOfWeight(gridError, gridNbpoint, gridInter, gridUnion, threshold, threshold_dice)
        nbwellRegister = sum(gridWeight[0,:])
        print('Nb slice well register :',nbwellRegister)
        

        
        for i_slice in range(nbSlice):
             
             okpre=False;okpost=False;nbSlice1=0;nbSlice2=0;dist1=0;dist2=0
             slicei=listSlice[i_slice]
             if gridWeight[0,i_slice]==0:
                 for i in range(i_slice,0,-2):
                     if listSlice[i].get_orientation()==listSlice[i_slice].get_orientation():
                         if gridWeight[0,i]==1 :
                             nbSlice1=i
                             dist1=np.abs(i_slice-nbSlice1)//2
                             okpre=True
                             break
                 for j in range(i_slice,nbSlice,2):
                    if gridWeight[0,j]==1:
                        if listSlice[j].get_orientation()==listSlice[i_slice].get_orientation():
                             nbSlice2=j
                             dist2=np.abs(i_slice-nbSlice2)//2
                             okpost=True
                             break
                 if okpre==True and okpost==True: #if there is two close slice well-register, we do a mean between them
                     Slice1=listSlice[nbSlice1];Slice2=listSlice[nbSlice2]
                     ps1=Slice1.get_parameters();ps2=Slice2.get_parameters();
                     MS1=rigidMatrix(ps1);MS2=rigidMatrix(ps2)
                     RotS1=MS1[0:3,0:3];RotS2=MS2[0:3,0:3]
                     TransS1=MS1[0:3,3];TransS2=MS2[0:3,3]
                     Rot=computeMeanRotation(RotS1,dist1,RotS2,dist2)
                     Trans=computeMeanTranslation(TransS1,dist1,TransS2,dist2)
                     Mtot=np.eye(4)
                     Mtot[0:3,0:3]=Rot;Mtot[0:3,3]=Trans
                     p=ParametersFromRigidMatrix(Mtot)
                     #debug_meanMatrix(Rot,Trans,p)

                     slicei.set_parameters(p)
                 elif okpre==True and okpost==False: #if there is only one close slice well-register, we take the parameters of this slice
                      Slice1=listSlice[nbSlice1]
                      ps1=Slice1.get_parameters()
                      slicei.set_parameters(ps1)
                 elif okpost==True and okpre==False: 
                      Slice2=listSlice[nbSlice2]
                      ps2=Slice2.get_parameters()
                      slicei.set_parameters(ps2)
                 
                 delta=5
                 initial_s = np.zeros((7,6))

                 #We do a new optimisation based on the new initialisation
                 d=delta  
                 x0=slicei.get_parameters()
                 xatol=0.01
                 costMse,costDice,gridError,gridNbpoint,gridInter,gridUnion,x_opt = SimplexOptimisation(d,xatol,x0,i_slice,listSlice,gridError,gridNbpoint,gridInter,gridUnion,initial_s,0)
                 slicei.set_parameters(x_opt)
   
        threshold_pre=threshold
        gridError,gridNbpoint,gridInter,gridUnion=computeCostBetweenAll2Dimages(listSlice)
        costMse=costFromMatrix(gridError,gridNbpoint)
        costDice=costFromMatrix(gridInter,gridUnion)
        updateResults(dicRes, gridError, gridNbpoint, gridInter, gridUnion, costMse, costDice, listSlice, nbSlice)
        
        
    return dicRes


def removeBadSlice(listSlice,gridInter,gridUnion):
    """
    return a list of bad-registered slices and their corresponding stack
    """
    removelist=[]
    for i_slice in range(len(listSlice)):
       i_dice=oneSliceCost(gridInter, gridUnion, i_slice)
       if i_dice<0.7:
           removelist.append((listSlice[i_slice].get_orientation(),listSlice[i_slice].get_index_slice()))
    return removelist
        
   
    
# def multi_start(listSlice,indexSlice):
    
#     delta=10
#     initial_s = np.zeros((7,6))

    
#     print('Begening of the algorithm')

#     boundsAngle=[-20,20]
#     boundsTranslation=[-20,20]
#     rangeAngle = boundsAngle[1]-boundsAngle[0]
#     rangeTranslation = boundsTranslation[1]-boundsTranslation[0]
#     minMse=10
#     bestParam=np.zeros(6)
    
#     for i in range(3):
        
#         slicei = listSlice[indexSlice]
#         x = 0
#         a1 = rd.random()*(rangeAngle) - (rangeAngle)/2
#         a2 = rd.random()*(rangeAngle) - (rangeAngle)/2
#         a3 = rd.random()*(rangeAngle) - (rangeAngle)/2
#         t1 = rd.random()*(rangeTranslation) - (rangeTranslation)/2
#         t2 = rd.random()*(rangeTranslation) - (rangeTranslation)/2
#         t3 = rd.random()*(rangeTranslation) - (rangeTranslation)/2
#         x_multistart = np.array([a1,a2,a3,t1,t2,t3])
#         slicei.set_parameters(x_multistart) 
#         nbSlice = len(listSlice)
#         #print('a1 :',a1,'a2 :',a2,'a3 :',a3,'t1 :',t1,'t2 : ',t2,'t3 : ',t3)        
#         gridError,gridNbpoint,gridInter,gridUnion = computeCostBetweenAll2Dimages(listSlice)       
               
#         x = slicei.get_parameters()
#         P0 = x
#         P1 = P0 + np.array([delta,0,0,0,0,0])
#         P2 = P0 + np.array([0,delta,0,0,0,0])
#         P3 = P0 + np.array([0,0,delta,0,0,0])
#         P4 = P0 + np.array([0,0,0,delta,0,0])
#         P5 = P0 + np.array([0,0,0,0,delta,0])
#         P6 = P0 + np.array([0,0,0,0,0,delta])
                                                                        
#         initial_s[0,:]=P0
#         initial_s[1,:]=P1
#         initial_s[2,:]=P2
#         initial_s[3,:]=P3
#         initial_s[4,:]=P4
#         initial_s[5,:]=P5
#         initial_s[6,:]=P6       
                                                                
#         NM = minimize(cost_fct,x,args=(indexSlice,listSlice,gridError,gridNbpoint,gridInter,gridUnion,np.ones((nbSlice,nbSlice)),10000,0),method='Nelder-Mead',options={"disp" : False, "maxiter" : 2000, "maxfev":1e4, "xatol" : 1e-1, "fatol" : 1e-2, "initial_simplex" : initial_s , "adaptive" :  False})
#         x_opt = NM.x
#         slicei.set_parameters(x_opt)
#         updateCostBetweenAllImageAndOne(indexSlice, listSlice, gridError, gridNbpoint,gridInter, gridUnion)
#         current_cost = costFromMatrix(gridError,gridNbpoint) 
                                        
#         se = sum(gridError[indexSlice,:]) + sum(gridError[:,indexSlice])
#         point =  sum(gridNbpoint[indexSlice,:]) + sum(gridNbpoint[:,indexSlice])
#         mse = se/point
#         #print('Mse of the slice : ',mse)
            
            
#         if mse<minMse:
#             bestParam=x_opt
#             minMse=mse
    

    
#     return minMse, bestParam