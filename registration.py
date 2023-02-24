# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 13:59:55 2021

@author: mercier
"""


import nibabel as nib 
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize
from scipy import stats
import time
from tools import createVolumesFromAlist, computeMeanRotation, computeMeanTranslation, ParametersFromRigidMatrix, rigidMatrix, denoising
from numba import jit
from multiprocessing import Pool
from functools import partial
import outliers_detection_intersection 
import display


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
def minLambda(lambdaPropo1,lambdaPropo2,inter='intersection'):
    
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
    if inter=='intersection':
        
        lambdaMin = np.zeros(2)
        
        lambdaMin[0] = max(min(lambdaPropo1),min(lambdaPropo2))
        lambdaMin[1] = min(max(lambdaPropo1),max(lambdaPropo2))
    
    else :
        
            if inter =='union':
                lambdaMin = np.zeros(2)
                lambdaMin[0] = min(min(lambdaPropo1),min(lambdaPropo2))
                lambdaMin[1] = max(max(lambdaPropo1),max(lambdaPropo2))
            else :
                print('inter must be intersection or union')
                
           
    
    
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

    lambdaMin = minLambda(lambdaPropo1,lambdaPropo2,'intersection')
      
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
    sliceimage1=slice1.get_slice().get_fdata();sliceimage2=slice2.get_slice().get_fdata();res=min(min(slice1.get_slice().header.get_zooms(),slice2.get_slice().header.get_zooms()))
    #res=1
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
            
        if slice2.ok==1:
                    
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
        if slice1.ok==1 :
            for i_slice2 in range(nbSlice):
                slice2=listSlice[i_slice2]
                if slice2.ok ==1:
                        if (i_slice1 > i_slice2):
                            if slice1.get_orientation() != slice2.get_orientation():
                                newError,commonPoint,inter,union=costLocal(slice1,slice2) #computed cost informations between two slices
                                gridError[i_slice1,i_slice2]=newError 
                                gridNbpoint[i_slice1,i_slice2]=commonPoint
                                gridInter[i_slice1,i_slice2]=inter
                                gridUnion[i_slice1,i_slice2]=union
                    
    return gridError,gridNbpoint,gridInter,gridUnion     



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
 

def global_optimisation(hyperparameters,listSlice,ablation):
    

    hyperparameters = np.asarray(hyperparameters,dtype=np.float64)

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
    


    nbslice=len(listSlice)
    set_o=np.zeros(nbSlice)
    

    grid_slices=np.array([gridError,gridNbpoint,gridInter,gridUnion])
    set_r=np.zeros(nbSlice)
    #First Step : sigma = 4.0 , d=b, x_opt, omega, 
    #epsilon = sqrt(6*(erreur)^2)
    if ablation=='no_gaussian':
        hyperparameters[5] = 0
    elif ablation=='no_dice':
        hyperparameters[4] = 0 
    
    print('ablation :', ablation)
    print('hyperparameters :', hyperparameters)
    
    if hyperparameters[5] != 0 :
        new_hyperparameters = np.array([hyperparameters[0],hyperparameters[1],hyperparameters[2],np.sqrt(6*hyperparameters[3]**2),hyperparameters[4],hyperparameters[5]])
        ge,gn,gi,gu,dicRes=algo_optimisation(new_hyperparameters,listSlice,set_o,set_r,grid_slices,dicRes)  
        grid_slices=np.array([ge,gn,gi,gu])
        set_r=np.zeros(nbSlice)
        #Second Step : sigma = 2.0, d=b/2, x_opt, omega
        #epsilon = sqrt(6*(erreur/2)^2)
        new_hyperparameters = np.array([hyperparameters[0]/2,hyperparameters[1],hyperparameters[2],np.sqrt(6*hyperparameters[3]**2/2),hyperparameters[4],hyperparameters[5]/2])
        ge,gn,gi,gu,dicRes=algo_optimisation(new_hyperparameters,listSlice,set_o,set_r,grid_slices,dicRes)  
        grid_slices=np.array([ge,gn,gi,gu])
    #Third step : sigma = 0.0, d=b/4, x_opt, omega
    #epsilon = sqrt(6*(erreur)^2/4)
    if hyperparameters[4] != 0 :
        new_hyperparameters = np.array([hyperparameters[0]/4,hyperparameters[1],hyperparameters[2],np.sqrt(6*hyperparameters[3]**2/4),hyperparameters[4],0])
        ge,gn,gi,gu,dicRes=algo_optimisation(new_hyperparameters,listSlice,set_o,set_r,grid_slices,dicRes)  
        grid_slices=np.array([ge,gn,gi,gu])
    #Fourth step : sigma = 0.0 , d=b/8, x_opt, omega
    #epsilon = sqrt(6*(erreur)^2/8)
    new_hyperparameters = np.array([hyperparameters[0]/8,hyperparameters[1],hyperparameters[2],np.sqrt(6*hyperparameters[3]**2/8),0,0])
    ge,gn,gi,gu,dicRes=algo_optimisation(new_hyperparameters,listSlice,set_o,set_r,grid_slices,dicRes)  
    grid_slices=np.array([ge,gn,gi,gu])
    
    union=np.zeros((nbSlice,nbSlice))
    intersection=np.zeros((nbSlice,nbSlice))
    
    nbSlice = len(listSlice)
    nb_slice_matrix = np.zeros((nbSlice,nbSlice)) 
    dist_union,dist_intersection,nb_slice_matrix = outliers_detection_intersection.compute_dice_error_for_each_slices(union,intersection,nb_slice_matrix,listSlice)
    ratio = np.abs(dist_union - dist_intersection)
    
    Num=ratio
    Denum=nb_slice_matrix
    nb_glob, ratio_glob  = display.indexGlobalMse(Num,Denum)
    diff=[e/n for (e,n) in zip(ratio_glob,nb_glob)]
    diff=np.array(diff)
    HistoDiff=diff[np.where((~np.isnan(diff) * ~np.isinf(diff)))]
    fit_alpha, fit_loc, fit_scale = stats.gamma.fit(HistoDiff)
    confidence = stats.gamma.interval(0.975,fit_alpha, fit_loc,fit_scale)
    t_inter = confidence[1]
    
    Num=ge
    Denum=gn
    point_glob, error_glob = display.indexGlobalMse(Num,Denum)
    mse=[e/n for (e,n) in zip(error_glob,point_glob)]
    mse=np.array(mse)
    HistoMSE=mse[np.where((~np.isnan(mse) * ~np.isinf(mse)))]
    fit_alpha, fit_loc, fit_scale = stats.gamma.fit(HistoMSE)
    confidence = stats.gamma.interval(0.975,fit_alpha, fit_loc,fit_scale)
    t_mse = confidence[1]
    
    
    
    
    mW = matrixOfWeight(ge, gn, union, intersection, listSlice,t_inter,t_mse)
    set_o = np.abs(np.ones(nbSlice)-mW[0,:])
    print("badly register : ",sum(set_o))
    ge,gn,gi,gu,dicRes=algo_optimisation(new_hyperparameters,listSlice,set_o,set_r,grid_slices,dicRes)  
    grid_slices=np.array([ge,gn,gi,gu])
    
      
    if ablation!='no_multistart':
        grid_slices,dicRes=bad_slices_correction(listSlice,new_hyperparameters,set_o,grid_slices,dicRes,t_inter,t_mse)
   
    
    rejectedSlices=removeBadSlice(listSlice, grid_slices, t_inter, t_mse)
    #rejectedSlices=[]
    
    
    return dicRes, rejectedSlices
    

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
    n=gridError.shape[0]
    for i in range(n): #compute between each slice and its orthogonal slices
        mse = sum(gridError[i,:]) + sum(gridError[:,i])
        point =  sum(gridNbpoint[i,:]) + sum(gridNbpoint[:,i])
        vectMse[i] = mse/point
    
    vectMse = vectMse[~np.isnan(vectMse)]                    
    valMse = np.median(vectMse)
    sortMse = np.sort(vectMse)
    q1 = np.quantile(sortMse,0.25)
    q3 = np.quantile(sortMse,0.75)
    iqr=q3-q1
    upper_fence = q3 + (1.5*iqr)

    #std = 1.4826*np.median(np.abs(vectMse-valMse)) #the distribution of MSE is approximately gaussien so we take the 95% intervall
    #threshold = np.mean(vectMse) + 6*std
    threshold = upper_fence
    
    return threshold


def MseThreshold(gridError,gridNbpoint):
    """
    Compute the threshold between well-register slices and bad-register slices
    
    Inputs 
    gridError : triangular matrix representing the square error between each slices
    gridNbpoint : triangular matrix representing the number of common point between each slices

    Ouptut 
    threshold : scalar, if the MSE of a slice is above this value, it is well-register, if not, it is badly-register
        
    """
    
    vectMse = np.zeros([gridError.shape[0]])
    n=gridError.shape[0]
    for i in range(n): #compute between each slice and its orthogonal slices
        mse = sum(gridError[i,:]) + sum(gridError[:,i])
        point =  sum(gridNbpoint[i,:]) + sum(gridNbpoint[:,i])
        vectMse[i] = mse/point
    
    vectMse = vectMse[~np.isnan(vectMse)]                    
    valMse = np.median(vectMse)
    sortMse = np.sort(vectMse)

    std = 1.4826*np.median(np.abs(vectMse-valMse)) #the distribution of MSE is approximately gaussien so we take the 95% intervall
    threshold = np.mean(vectMse) + 6*std
    
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

    Outputs : 
        
    x_opt : corrected parameters
    finish_val : boolean, indicate if the slice is already register or not

    """
    delta = d 
    slicei=listSlice[nImage]
    x0=slicei.get_parameters()
 
    previous_param=x0
    finish_val=finish[nImage]
    if finish_val==0:
        costMse,costDice,gridError,gridNbpoint,gridInter,gridUnion,x_opt=SimplexOptimisation(delta,xatol, x0, nImage, listSlice, gridError, gridNbpoint, gridInter, gridUnion, initial_s, lamb)
        current_param=x_opt
        

        

        if np.linalg.norm(previous_param-current_param)<1e-1:
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



def bad_slices_correction(listSlice,hyperparameters,set_o,grid_slices,dicRes,t_inter,t_mse):
    """
    This function aims to correct mis-registered slices

    Parameters
    ----------
    listSlice : set of the slices, from the three volumes
    hyperparameters : parameters for optimisation, it includes : simplex initial size, xatol, fatol, epsilon, gaussian parameter, lamnda for dice
    set_o : set of outliers or mis-registered slices
    grid_slices : matrix of cost
    dicRes : results save for representation

    Returns
    -------
    grid_slices : matrix of cost
    dicRes : results save for representation

    """
    
    gridError = grid_slices[0]
    gridNbpoint = grid_slices[1]
    gridInter = grid_slices[2]
    gridUnion = grid_slices[3]
    before_correction = sum(set_o)
    while True:
        nbSlice = len(listSlice) #number of slices total
        for i_slice in range(nbSlice):
                 
            okpre=False;okpost=False;nbSlice1=0;nbSlice2=0;dist1=0;dist2=0
            slicei=listSlice[i_slice]
            if set_o[i_slice]==1:
                for i in range(i_slice,0,-2):
                    if listSlice[i].get_orientation()==listSlice[i_slice].get_orientation(): #check if the previous slice is from the same volume
                        if set_o[i]==0 : #if the previous slice is well registered 
                            nbSlice1=i
                            dist1=np.abs(i_slice-nbSlice1)//2
                            okpre=True
                            break
                for j in range(i_slice,nbSlice,2):
                    if set_o[j]==0: #if the previous slice is well registered
                        if listSlice[j].get_orientation()==listSlice[i_slice].get_orientation(): 
                             nbSlice2=j
                             dist2=np.abs(i_slice-nbSlice2)//2
                             okpost=True
                             break
                if okpre==True and okpost==True: #if there is two close slice well-register, we do a mean between them
                     Slice1=listSlice[nbSlice1];Slice2=listSlice[nbSlice2]
                     ps1=Slice1.get_parameters().copy();ps2=Slice2.get_parameters().copy();
                     MS1=rigidMatrix(ps1);MS2=rigidMatrix(ps2)
                     RotS1=MS1[0:3,0:3];RotS2=MS2[0:3,0:3]
                     TransS1=MS1[0:3,3];TransS2=MS2[0:3,3]
                     Rot=computeMeanRotation(RotS1,dist1,RotS2,dist2)
                     Trans=computeMeanTranslation(TransS1,dist1,TransS2,dist2)
                     Mtot=np.eye(4)
                     Mtot[0:3,0:3]=Rot;Mtot[0:3,3]=Trans
                     p=ParametersFromRigidMatrix(Mtot)
          
    
                     slicei.set_parameters(p)
                elif okpre==True and okpost==False: #if there is only one close slice well-register, we take the parameters of this slice
                     Slice1=listSlice[nbSlice1]
                     ps1=Slice1.get_parameters().copy()
                     slicei.set_parameters(ps1)
                elif okpost==True and okpre==False: 
                     Slice2=listSlice[nbSlice2]
                     ps2=Slice2.get_parameters().copy()
                     slicei.set_parameters(ps2)
                     
    
    
                #We do a multistart optimisation based on the new initialisation
                x0=slicei.get_parameters()
                multistart=np.random.rand(5)*20 - 10
                multistart = np.concatenate(([0],multistart))
                with Pool() as p:
                     tmpfun=partial(multi_start,hyperparameters,i_slice,listSlice,grid_slices,set_o,x0)
                     res=p.map(tmpfun,multistart)
                p=[p[0] for p in res]
                print(p)
                mincost = min(p)
                x=[x[1] for x in res]
                x_opt = x[p.index(mincost)]
                print('i_slice',i_slice)
                print(x_opt)
                print('mincost :',mincost)
                slicei.set_parameters(x_opt)
                
                
                gridError,gridNbpoint,gridInter,gridUnion=computeCostBetweenAll2Dimages(listSlice)     
                costMse=costFromMatrix(gridError,gridNbpoint)
                print('cost :', costMse)
                costDice=costFromMatrix(gridInter,gridUnion)
        
        union=np.zeros((nbSlice,nbSlice))
        intersection=np.zeros((nbSlice,nbSlice))
        mW = matrixOfWeight(gridError, gridNbpoint, union, intersection, listSlice,t_inter,t_mse)
        new_set_o = np.abs(np.ones(nbSlice)-mW[0,:])
        print("badly register : ", sum(new_set_o))
        
        if np.all(new_set_o == set_o):
            break
            
        set_o = new_set_o.copy()
    
    set_o = new_set_o
    after_correction = sum(set_o)
    saved = before_correction - after_correction
    print('slices saved with multistart :', saved)
    ge,gn,gi,gu=computeCostBetweenAll2Dimages(listSlice)
    costMse=costFromMatrix(gridError,gridNbpoint)
    costDice=costFromMatrix(gridInter,gridUnion)
    updateResults(dicRes, ge, gn, gi, gu, costMse, costDice, listSlice, nbSlice)
    grid_slices = np.array([ge,gn,gi,gu])
        
    return grid_slices,dicRes


def multi_start(hyperparameters,i_slice,listSlice,grid_slices,set_o,x0,valstart):
    """
    Function to try different initial position for optimisation. 

    Parameters
    ----------
    hyperparameters : parameters for optimisation : simplex size, xatol, fatol, epsilon, gauss, lamb
    i_slice : slice we want to correct
    listSlice : set of slices
    grid_slices : matrix costs
    set_o : set of outliers
    x0 : initial postion of the slice
    valstart : value for mutlistart

    Returns
    -------
    cost :
        cost after optimisation
    
    x_opt :
        best parameters of slices obtained with optimisation

    """
    x0=x0+valstart
    opti_res = SimplexOptimisation(x0,hyperparameters,listSlice,grid_slices,set_o,i_slice)
    cost=opti_res[0]
    x_opt=opti_res[3]
    
    return cost,x_opt


def matrixOfWeight(gridError,gridNbpoint,intersection,union,listSlice,t_inter,t_mse):
    """
    Compute a binary matrix which represent if a slice is well-register of bad-register. Each column of the matrix represent a slice

    Inputs 
    gridError : triangular matrix representing the square error between each slices
    gridNbpoint : triangular matrix representing the number of common point between each slices
    threshold : scalar, if the MSE of a slice is above this value, it is well-register, if not, it is badly-register
        
    Ouptuts 
    Weight : matrix of registered (1) and outliers (0) slices
    """
    X,Y = gridError.shape
    Weight = np.ones((X,Y))
    nbSlice = len(listSlice)
    nb_slice_matrix = np.zeros((nbSlice,nbSlice)) 
    dist_union,dist_intersection,nb_slice_matrix = outliers_detection_intersection.compute_dice_error_for_each_slices(union,intersection,nb_slice_matrix,listSlice)
    ratio = np.abs(dist_union - dist_intersection)
    
    listSliceError = []
    for i_slice in range(len(listSlice)):
        
        slicei = listSlice[i_slice]
        slice_index = slicei.get_index_slice()
        orientation = slicei.get_orientation()
        
        ErrorSlicei = outliers_detection_intersection.ErrorSlice(orientation,slice_index)
        nbpoint_binary_map = sum(nb_slice_matrix[i_slice,:]) + sum(nb_slice_matrix[:,i_slice])
        error_distance = sum(ratio[i_slice,:]) + sum(ratio[:,i_slice])
        error_distance_mean = error_distance/nbpoint_binary_map
        ErrorSlicei.set_inter(error_distance_mean)
        #print(i_slice,error_distance_mean)
        
        mse_slice = (sum(gridError[i_slice,:]) + sum(gridError[:,i_slice]))/(sum(gridNbpoint[i_slice,:]) + sum(gridNbpoint[:,i_slice]))
        #print(i_slice,mse_slice)
         
        ErrorSlicei.set_mse(mse_slice)
        mask = slicei.get_mask()
        pmask = outliers_detection_intersection.mask_proportion(mask)
        ErrorSlicei.set_mask_proportion(pmask)
        
        listSliceError.append(ErrorSlicei)
    
    listvolumeSliceError = outliers_detection_intersection.createVolumesFromAlistError(listSliceError)
    
    nb_stack = len(listvolumeSliceError)
    mask_size = np.zeros((nbSlice,nb_stack))
    
    for i_stack in range(nb_stack):
        stack  = listvolumeSliceError[i_stack]
        nb_error = len(stack)
        for i_slice in range(nb_error):
            SErrori = stack[i_slice]
            mask_size[i_slice,i_stack] = SErrori.get_mask_proportion()*100
    
    
    for i_stack in range(nb_stack):
        
        stack  = listvolumeSliceError[i_stack]
        nb_error = len(stack)
        
        vect = mask_size[:,i_stack]
        vectMask = vect[vect>0]

       

        mean = np.quantile(vectMask,0.1)

       
        t = mean 
        
        for i_slice in range(nb_error):
            SErrori = stack[i_slice]
            pmask = SErrori.get_mask_proportion() 
            if 100*pmask < t:
                SErrori.set_bords(True) 

    for i_stack in range(nb_stack):
        stack  = listvolumeSliceError[i_stack]
        nb_error = len(stack)
        for i_slice in range(nb_error):
            SErrori = stack[i_slice]
            if SErrori.edge() == True:
                if SErrori.get_inter() > t_inter or np.isnan(SErrori.get_inter()) or np.isnan(SErrori.get_mse()) or np.isinf(SErrori.get_inter()) or np.isinf(SErrori.get_mse()):
                    index_list = outliers_detection_intersection.get_index(SErrori.get_orientation(),SErrori.get_index(),listSlice)
                    if index_list != -1 :
                        Weight[:,index_list] = 0
            else :    
                if SErrori.get_mse() > t_mse or np.isnan(SErrori.get_mse()) or np.isnan(SErrori.get_inter()) or np.isinf(SErrori.get_inter()) or np.isinf(SErrori.get_mse()):
                    index_list = outliers_detection_intersection.get_index(SErrori.get_orientation(),SErrori.get_index(),listSlice)
                    if index_list != -1 :
                        Weight[:,index_list] = 0

    return Weight



def removeBadSlice(listSlice,grid_slices,t_inter,t_mse):
    """
    return a list of bad-registered slices and their corresponding stack
    """
    removelist=[]
    ge = grid_slices[0,:,:]
    gn = grid_slices[1,:,:]
    nbSlice=len(listSlice)
    union=np.zeros((nbSlice,nbSlice))
    intersection=np.zeros((nbSlice,nbSlice))
    mW = matrixOfWeight(ge, gn, union, intersection, listSlice,t_inter,t_mse)
    for i_slice in range(len(listSlice)):
      if mW[0,i_slice]==0:
          removelist.append((listSlice[i_slice].get_orientation(),listSlice[i_slice].get_index_slice()))
   
    
    return removelist
        


def cost_from_matrix(grid_numerator,grid_denumerator,set_o,i_slice):
    """
    Function to compute the cost, either mse or dice. Cost is computed only on well registered slices and depend on the slice we want ot make the optimisation on.

    Parameters
    ----------
    grid_numerator : grid_error or grid_intersection
    grid_denumerator : grid_nbpoint or grid_union
    set_o : outliers slices
    i_slice :  slice considered in the optmisation

    Returns
    -------
    cost : mse or dice
    """
    
    nbslice,nbslice = grid_numerator.shape
    
    grid_numerator_no_o = grid_numerator.copy()
    grid_denumerator_no_o = grid_denumerator.copy()
    
    set_outliers = np.abs(1-set_o.copy())
    
    set_outliers[i_slice]=1

    grid_numerator_no_o=[np.sqrt(x*y) for x,y in zip(grid_numerator*set_outliers,np.transpose(np.transpose(grid_numerator)*set_outliers))]
    
    grid_denumerator_no_o=[np.sqrt(x*y) for x,y in zip(grid_denumerator*set_outliers,np.transpose(np.transpose(grid_denumerator)*set_outliers))]
     
    numerator = np.sum(grid_numerator_no_o)

    denumerator = np.sum(grid_denumerator_no_o)
    
    cost = numerator/denumerator
  
  
    return cost
            

def cost_fct(x,i_slice,listSlice,grid_slices,set_o,lamb):
    """
    function we want to minimize. 

    Parameters
    ----------
    x : initial parameters of the slice, parameters of the optimisation
    i_slice : slice of interset
    listSlice : set of slices
    grid_slices : matrix of cost
    set_o : set of outliers
    lamb : coefficient of the dice

    Returns
    -------
    cost : cost, composed of the mse and dice

    """
    
    ge = grid_slices[0,:,:]
    gn = grid_slices[1,:,:]
    gi = grid_slices[2,:,:]
    gu = grid_slices[3,:,:]
    
    slicei = listSlice[i_slice]
    slicei.set_parameters(x)
    #print(x)
    

    updateCostBetweenAllImageAndOne(i_slice,listSlice,ge,gn,gi,gu)
    
    grid_error=ge
    grid_nbpoint=gn
    grid_intersection=gi
    grid_union=gu
    
    grid_slices = np.array([grid_error,grid_nbpoint,grid_intersection,grid_union])
   
    
    mse = cost_from_matrix(grid_error,grid_nbpoint,set_o,i_slice)
    dice = cost_from_matrix(grid_intersection,grid_nbpoint,set_o,i_slice)

    cost = mse - lamb*dice

    
    return cost

def cost_optimisation_step(hyperparameters,listSlice,grid_slices,set_r,set_o,n_image):
    """
    function with optimise the cost function for one slice

    Parameters
    ----------
    hyperparameters : parameters for optimisation (initial simplex size, xatol,fatol,epsilon, gauss, lamb)
    listSlice : set of slices
    grid_slices : matrix of slices
    set_r : set of slices to register
    set_o : set of outliers
    n_image : slice of interest

    Returns
    -------
    x_opt : best parameters for the slice
    registration_state : 1 if the slice is well register after optimisation, 0 else.

    """
    
    epsilon = hyperparameters[3]
    


    slicei=listSlice[n_image]
    x0=slicei.get_parameters()
    previous_param=x0
    
    registration_state = set_r[n_image]
 
    
    #if registration_state == 0:
    costMse,costDice,grid_slices,x_opt=SimplexOptimisation(x0,hyperparameters,listSlice,grid_slices,set_o,n_image)
    current_param=x_opt
    #print('previous',previous_param,'current',current_param,epsilon)
    if np.linalg.norm(previous_param-current_param)<epsilon:
        registration_state=1
        print(n_image,'slice ok')
    #else:
    #    x_opt=x0

    return x_opt,registration_state      
    
def algo_optimisation(hyperparameters,listSlice,set_o,set_r,grid_slices,dicRes):
    
    """
    function to optimise cost on all slices
    
    hyperparameters: parameters for optimisation (simplex initial size, xatol, fatol, epsilon, gauss, lamb)
        
    listSlice: set of slice
        
    set_o: set of outliers slices
        
    dicRes: save results for representation
    """
    
    
    max_extern = 10
    max_intern = 10
    
    it_extern = 0
    it_intern = 0
    
    images,mask=createVolumesFromAlist(listSlice)
                                       
    nb_images=len(images)         
    nbSlice = len(listSlice)    
    
    blur = hyperparameters[5]
    print('blur :', blur)
    print('Hehehe')
    listSliceblur = denoising(listSlice.copy(),blur)
    ge,gn,gi,gu=computeCostBetweenAll2Dimages(listSliceblur)
    grid_slices=np.array([ge,gn,gi,gu])
    
    while (it_intern > 1 or it_extern == 0) and it_extern < max_extern:
        
        print("it_extern:",it_extern)
        print("it_intern:",it_intern)
        
        set_to_register=set_r.copy() #at the begenning, we need to register all the slices
        set_to_register_pre = set_to_register.copy()
        it_intern = 0
        while set_to_register.all()!=1 and it_intern < max_intern and (not sum(set_to_register)==sum(set_to_register_pre) or sum(set_to_register)==0) :
            #avoid that the algorithm continue because there is one or two slices not register
            set_to_register_pre = set_to_register.copy()
            index_pre=0
            for n_image in range(nb_images):
                start = time.time()
                nbSliceImageN=len(images[n_image])
                randomIndex= np.arange(nbSliceImageN) 
                index=randomIndex+index_pre
                eval_index=np.where(set_to_register[index]==0)[0]
                eval_index=eval_index+index_pre
                
                
                        
                with Pool(processes=12) as p: #all slices from the same stacks are register together (in the limit of the number of CPU on your machine)
                    tmpfun=partial(cost_optimisation_step,hyperparameters,listSliceblur,grid_slices,set_to_register,set_o) 
                    res=p.map(tmpfun,eval_index)
                    
                                              
                for m,i_slice in enumerate(eval_index): #update parameters once the registration is done
                    print(m,i_slice)
                    listSliceblur[i_slice].set_parameters(res[m][0])
                    set_to_register[i_slice]=res[m][1]
                
                
                ge,gn,gi,gu=computeCostBetweenAll2Dimages(listSliceblur)
                grid_slices = np.array([ge,gn,gi,gu])
                costMse=costFromMatrix(ge,gn)
                costDice=costFromMatrix(gi,gn)
                print('mse:',costMse)
                index_pre=index_pre+len(images[n_image])
            
            it_intern+=1
            print('well registered :',np.sum(set_to_register>0))
            
        

        print('delta :',  hyperparameters[0], 'iteration :', it_extern, 'final MSE :', costMse, 'final DICE :', costDice)
        it_extern+=1 
        end = time.time()
        elapsed = end - start
        
        for i_slice in range(nbSlice):
            listSlice[i_slice].set_parameters(listSliceblur[i_slice].get_parameters())
                        
        ge,gn,gi,gu=computeCostBetweenAll2Dimages(listSlice)
        costMse=costFromMatrix(ge, gn)
        costDice=costFromMatrix(gi,gu)
        updateResults(dicRes,ge,gn,gi,gu,costMse,costDice,listSlice,nbSlice)
        print('MSE: ', costMse)
        print('Dice: ', costDice) 
        print(f'Temps d\'exécution : {elapsed}') 
        
        #it_extern+=1
    
    return ge,gn,gi,gu,dicRes

def SimplexOptimisation(x0,hyperparameters,listSlice,grid_slices,set_o,i_slice):
    
    """
    Implementation of the simplex optimisation, for on slice,  using scipy
    
    x0 : initial parameters of the slice
    
    hyperparameters : parameters used for optimisation
    
    listSlice : set of slices
    
    grid_slices : matrix of cost
    
    set_o : set of outliers slices
    
    i_slice : slice of interest
    
    """
    
    
    
    slicei=listSlice[i_slice]

    slicei.set_parameters(x0)
    delta=hyperparameters[0]
    xatol=hyperparameters[1]
    fatol=hyperparameters[2]
    lamb=hyperparameters[4]

        
    P0 = x0 #create the initial simplex
    P1 = P0 + np.array([delta,0,0,0,0,0])
    P2 = P0 + np.array([0,delta,0,0,0,0])
    P3 = P0 + np.array([0,0,delta,0,0,0])
    P4 = P0 + np.array([0,0,0,delta,0,0])
    P5 = P0 + np.array([0,0,0,0,delta,0])
    P6 = P0 + np.array([0,0,0,0,0,delta])

    initial_s = np.zeros((7,6))
    initial_s[0,:]=P0
    initial_s[1,:]=P1
    initial_s[2,:]=P2
    initial_s[3,:]=P3
    initial_s[4,:]=P4
    initial_s[5,:]=P5
    initial_s[6,:]=P6
                                           
    #X,Y = grid_slices[0,:,:].shape

    NM = minimize(cost_fct,x0,args=(i_slice,listSlice,grid_slices,set_o,lamb),method='Nelder-Mead',options={"disp" : True, "maxiter" : 2000, "maxfev":1e4, "xatol" : xatol, "initial_simplex" : initial_s , "adaptive" :  True})
        #optimisation of the cost function using the simplex method                                    
        
    x_opt = NM.x #best parameter obtains after
    print(NM.message)
    
    listSlice[i_slice].set_parameters(x_opt)
    
    #slicei.set_parameters(x_opt)
    ge=grid_slices[0,:,:]
    gn=grid_slices[1,:,:]
    gi=grid_slices[2,:,:]
    gu=grid_slices[3,:,:]

    updateCostBetweenAllImageAndOne(i_slice,listSlice,ge,gn,gi,gu)
    grid_slices = np.array([ge,gn,gi,gu])
    costMse=cost_from_matrix(ge,gn,set_o,i_slice) #MSE after optimisation
    #print('i_slice :', i_slice, 'cost :',costMse)
    costDice=cost_from_matrix(gi,gu,set_o,i_slice) #Dice after optimisation

        
    return costMse,costDice,grid_slices,x_opt