# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 13:59:55 2021

@author: mercier
"""


from nibabel import Nifti1Image
import numpy as np
from numpy import concatenate, copy, transpose, mean,sqrt,asarray,array, cross, std,ascontiguousarray,float_,equal, where, abs, shape,nan,reshape,isinf,eye,logical_or,ones,all,linspace,sum,ceil,isnan,zeros, logical_and
from numpy.linalg import norm,inv
from numpy.random import shuffle
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize
from scipy import stats
from time import perf_counter
from tools import createVolumesFromAlist, computeMeanRotation, computeMeanTranslation, ParametersFromRigidMatrix, rigidMatrix, denoising, computeMaxVolume
from numba import jit,njit
from multiprocessing import Pool
from functools import partial
import display
import scipy
import pickle
from outliers_detection_intersection import ErrorSlice, createVolumesFromAlistError
from scipy.optimize import Bounds


loaded_model = pickle.load(open('my_new_model.pickle', "rb"))

def commonSegment2(sliceimage1,M1,sliceimage2,M2,res):
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
    ok=int(ok[0])

    
    if ok<1: #if there is no intersection lines (the 2 planes are parralel) it is useless to compute the intersectionSegment
        return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)
        
    #sliceimage1=Slice1.get_slice().get_fdata()
    lambdaPropo1,ok=intersectionSegment(sliceimage1,M1,coeff,pt) #if there is no intersection segment (the line of intersection is outisde of the image or on a corner), it useless to compute a common segment
    ok=int(ok[0])
   
    if ok<1:
        return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)
    
    #sliceimage2=Slice2.get_slice().get_fdata()
    lambdaPropo2,ok=intersectionSegment(sliceimage2,M2,coeff,pt)
    ok=int(ok[0])
    
    if ok<1:
        return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)

    lambdaMin = minLambda(lambdaPropo1,lambdaPropo2,'intersection')

      
    if lambdaMin[0]==lambdaMin[1]: #the segment is nul, there is no intersection
        return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)
        
    point3D = zeros((3,2))
    
    point3D[0:3,0] = lambdaMin[0] * coeff + pt #Point corresponding to the value of lambda
    point3D[0:3,1] = lambdaMin[1] * coeff + pt
    
    point3D = concatenate((point3D,array([[1,1]])))
    
    pointImg1 = zeros((4,2))
    pointImg1[3,:] = ones((1,2))
    
    
    pointImg2 = zeros((4,2))
    pointImg2[3,:] = ones((1,2))
    
    
    pointImg1 = ascontiguousarray(inv(M1)) @ ascontiguousarray(point3D)

    pointImg2 = ascontiguousarray(inv(M2)) @ ascontiguousarray(point3D) 

    
    distance1 = norm(pointImg1[0:2,0] - pointImg1[0:2,1]) #distance between two points on the two images
    distance2 = norm(pointImg2[0:2,0] - pointImg2[0:2,1]) 


    #res = min(Slice1.get_slice().header.get_zooms())
      #the smaller resolution of a voxel
        
    if res<0: #probmem with the resolution of the image
        return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)
        
    if max(distance1,distance2)<1: #no pixel in commun
        return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)
       
    nbpoint = int(round(max(distance1,distance2)+1)/res) #choose the max distance and divide it by the smaller resolution 

    return pointImg1[0:2],pointImg2[0:2],(nbpoint)*ones((2,2),dtype=float_),ones((2,2),dtype=float_)



@jit(nopython=True,fastmath=True)
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
    n1 = cross(M1[0:3,0],M1[0:3,1]) 
    if norm(n1)<1e-6: #no division by 0, case M11 // M12 (normally that souldn't be the case)
        return array([float_(0.0)]),array([float_(0.0)]),array([float_(0.0)])
    n1norm = n1/norm(n1)
    n1 = n1norm
    t1 = M1[0:3,3]

    
    n2 = cross(M2[0:3,0],M2[0:3,1]) 
    if norm(n2)<1e-6: #no division by 0, case M21 // M22 (normally that souldn't be the case)
        return array([float_(0.0)]),array([float_(0.0)]),array([float_(0.0)])
    n2norm = n2/norm(n2)
    n2 = n2norm
    t2 = M2[0:3,3]

    
    alpha = ascontiguousarray(n1) @ ascontiguousarray(n2) #if the vector are colinear alpha will be equal to one (since n1 and n2 are normalized), can happend if we consider two parralel slice
    beta =  ascontiguousarray(n1) @ ascontiguousarray(t1)
    gamma = ascontiguousarray(n2) @ ascontiguousarray(t2)   
    
    if abs((1 - alpha*alpha))<1e-6: #if the vector are colinear, there is no intersection
        return array([float_(0.0)]),array([float_(0.0)]),array([float_(0.0)])
    a = 1/(1 - alpha*alpha)
    g = a*(beta - alpha*gamma)
    h = a*(gamma - alpha*beta)

    #line equation
    coeff = cross(n1,n2)
    pt = g*n1 + h*n2

    return coeff, pt, array([float_(1.0)])
 
@jit(nopython=True,fastmath=True)  
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
    Minv = inv(M)
    rinv = inv(M[0:3,0:3])
    n = ascontiguousarray(rinv) @ ascontiguousarray(coeff)
    pt = concatenate((pt,array([1])))
    ptimg = ascontiguousarray(Minv) @ ascontiguousarray(pt)
   
    a = -n[1]
    b = n[0]
    c = -(a*ptimg[0] + b*ptimg[1])
    
    
    #Intersection with the plan
    intersection = zeros((4,2)) #2 points of intersection of coordinates i,j,k 
    intersection[3,:] = ones((1,2))
    width = sliceimage.shape[0]-1
    height = sliceimage.shape[1]-1
 

    indice=0
    #The intersection on a corner are considered only once
    
    if (abs(a)>1e-10): #if a==0, the division by zeros in not possible, in this case we have only two intersection possible : 
           
        i=(-c/a); j=0
        
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
        return array([float_(0.0)]),array([float_(0.0)])
        
    #Compute the intersection point coordinates in the 3D space
    interw = zeros((4,2)) #2 points of intersection, with 3 coordinates x,y,z
    interw[3,:] = ones((1,2))
    interw = ascontiguousarray(M) @ ascontiguousarray(intersection) 
    
    interw[0:3,0] = interw[0:3,0] - pt[0:3]
    interw[0:3,1] = interw[0:3,1] - pt[0:3]
    
    squareNorm = ascontiguousarray(coeff) @ ascontiguousarray(coeff.transpose())
    lambdaPropo = ascontiguousarray((((1/squareNorm) * coeff.transpose())) @ ascontiguousarray(interw[0:3,:])) 

    
    return lambdaPropo,array([float_(1.0)])


    
@jit(nopython=True,fastmath=True)    
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
        
        lambdaMin = zeros(2)
        
        lambdaMin[0] = max(np.min(lambdaPropo1),np.min(lambdaPropo2))
        lambdaMin[1] = min(np.max(lambdaPropo1),np.max(lambdaPropo2))
    
    else :
        
            if inter =='union':
                lambdaMin = zeros(2)
                lambdaMin[0] = min(np.min(lambdaPropo1),np.min(lambdaPropo2))
                lambdaMin[1] = max(np.max(lambdaPropo1),np.max(lambdaPropo2))
            else :
                print('inter must be intersection or union')
                
           
    
    
    return lambdaMin #Return 2 values of lambda that represent the common segment between the 2 slices

#@jit(nopython=True)
def commonSegment(slice1,M1,slice2,M2,res):
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

    sliceimage1 = slice1.get_slice().get_fdata()
    sliceimage2 = slice2.get_slice().get_fdata()

    coeff,pt,ok=intersectionLineBtw2Planes(M1,M2)
    ok=int(ok[0])

    
    if ok<1: #if there is no intersection lines (the 2 planes are parralel) it is useless to compute the intersectionSegment
        return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)
        
    #sliceimage1=Slice1.get_slice().get_fdata()
    lambdaPropo1,ok=intersectionSegment(sliceimage1,M1,coeff,pt) #if there is no intersection segment (the line of intersection is outisde of the image or on a corner), it useless to compute a common segment
    ok1=int(ok[0])
   
    #if ok<1:
    #    return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)
    
    #sliceimage2=Slice2.get_slice().get_fdata()
    lambdaPropo2,ok=intersectionSegment(sliceimage2,M2,coeff,pt)
    ok2=int(ok[0])
    
    if ok1<1 and ok2<1:
        return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)
    if ok1<1 or ok2<1:
        
        if ok1<1:
            lambdaPropo=lambdaPropo2
        else:
            lambdaPropo=lambdaPropo1
        point3D = zeros((3,2))
            
        point3D[0:3,0] = lambdaPropo[0] * coeff + pt #Point corresponding to the value of lambda
        point3D[0:3,1] = lambdaPropo[1] * coeff + pt
            
        point3D = concatenate((point3D,array([[1,1]])))
            
        pointImg1 = zeros((4,2))
        pointImg1[3,:] = ones((1,2))
                        
        pointImg2 = zeros((4,2))
        pointImg2[3,:] = ones((1,2))
                        
        pointImg1 = ascontiguousarray(inv(M1)) @ ascontiguousarray(point3D) #there is a point, it simply not belong to the image but it's in the plan

        pointImg2 = ascontiguousarray(inv(M2)) @ ascontiguousarray(point3D)

        if not equal(pointImg1[0:2,0],pointImg1[0:2,1]).all() :
            distance1 = norm(pointImg1[0:2,0] - pointImg1[0:2,1])
        else :
            distance1=0
 
        if not equal(pointImg2[0:2,0],pointImg2[0:2,1]).all():
            distance2 = norm(pointImg2[0:2,0]- pointImg2[0:2,1])
        else:
            distance2=0
        if distance1 + distance2 == 0:
            return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)

        nbpoint=int((round(distance1+distance2)+1)/res)
    
        val1,index1,nbpointSlice1=sliceProfil(slice1,pointImg1,nbpoint)
        val2,index2,nbpointSlice2=sliceProfil(slice2,pointImg2,nbpoint)
 
        val1_res = concatenate((val1[index1],zeros(nbpointSlice2)))
        val2_res = concatenate((zeros(nbpointSlice1),val2[index2]))
        val1 = val1_res
        val2 = val2_res
        index1_res = concatenate((index1[index1],zeros(nbpointSlice2)))
        index2_res = concatenate((zeros(nbpointSlice1),index2[index2]))
        index1 = index1_res
        index2 = index2_res
        
        nbpoint=nbpointSlice1+nbpointSlice2
        if nbpoint==0:
            return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)
        
    elif max(lambdaPropo1)>min(lambdaPropo2) and max(lambdaPropo2)>min(lambdaPropo1):
            
            lambdaMin = minLambda(lambdaPropo1,lambdaPropo2,'union')
        
            if lambdaMin[0]==lambdaMin[1]: #the segment is nul, there is no intersection
                return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)
                
            point3D = zeros((3,2))
            
            point3D[0:3,0] = lambdaMin[0] * coeff + pt #Point corresponding to the value of lambda
            point3D[0:3,1] = lambdaMin[1] * coeff + pt
            
            point3D = concatenate((point3D,array([[1,1]])))
            
            pointImg1 = zeros((4,2))
            pointImg1[3,:] = ones((1,2))
            
            
            pointImg2 = zeros((4,2))
            pointImg2[3,:] = ones((1,2))
           
            
            pointImg1 = ascontiguousarray(inv(M1)) @ ascontiguousarray(point3D)

            pointImg2 = ascontiguousarray(inv(M2)) @ ascontiguousarray(point3D) 
            
            distance1 = norm(pointImg1[0:2,0] - pointImg1[0:2,1]) #distance between two points on the two images
            distance2 = norm(pointImg2[0:2,0] - pointImg2[0:2,1]) 

            #res = min(Slice1.get_slice().header.get_zooms())
            #the smaller resolution of a voxel
                
            if res<0: #probmem with the resolution of the image
                return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)        
            if max(distance1,distance2)<1: #no pixel in commun
                return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)
            
            nbpoint = int(round(max(distance1,distance2)+1)/res) #choose the max distance and divide it by the smaller resolution 

            newError=0; commonPoint=0; inter=0; union=0; ncc_var=-1; mse_coupe=(-1,-1)
            #print(ok)
            
            if ok>0:
                val1,index1,nbpointSlice1=sliceProfil(slice1, pointImg1, nbpoint)  #profile in slice 1
                #print(index1)
                val2,index2,nbpointSlice2=sliceProfil(slice2, pointImg2, nbpoint)  #profile in slice 2
                #print(index2)
    else:

 
        point3D = zeros((3,2))
            
        point3D[0:3,0] = lambdaPropo1[0] * coeff + pt #Point corresponding to the value of lambda
        point3D[0:3,1] = lambdaPropo1[1] * coeff + pt
            
        point3D = concatenate((point3D,array([[1,1]])))
            
        pointImg1 = zeros((4,2))
        pointImg1[3,:] = ones((1,2))
                        
        pointImg2 = zeros((4,2))
        pointImg2[3,:] = ones((1,2))
                        
        pointImg1 = ascontiguousarray(inv(M1)) @ ascontiguousarray(point3D)

        pointImg2 = ascontiguousarray(inv(M2)) @ ascontiguousarray(point3D)


        if not equal(pointImg1[0:2,0],pointImg1[0:2,1]).all() :
            distance1 = norm(pointImg1[0:2,0] - pointImg1[0:2,1])
        else :
            distance1=0
 
        if not equal(pointImg2[0:2,0],pointImg2[0:2,1]).all():
            distance2 = norm(pointImg2[0:2,0]- pointImg2[0:2,1])
        else:
            distance2=0
        if distance1 + distance2 == 0:
            return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)

        nbpoint=int((round(distance1+distance2)+1)/res)
    
        val1,index1,nbpointSlice1=sliceProfil(slice1,pointImg1,nbpoint)
        val2,index2,nbpointSlice2=sliceProfil(slice2,pointImg2,nbpoint)
 
        val1_res = concatenate((val1[index1],zeros(nbpointSlice2)))
        val2_res = concatenate((zeros(nbpointSlice1),val2[index2]))
        val1 = val1_res
        val2 = val2_res
        index1_res = concatenate((index1[index1],zeros(nbpointSlice2)))
        index2_res = concatenate((zeros(nbpointSlice1),index2[index2]))
        index1 = index1_res
        index2 = index2_res
        
        nbpoint=nbpointSlice1+nbpointSlice2
        if nbpoint==0:
            return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)


    return pointImg1,pointImg2,val1,val2,index1,index2,(nbpoint)*ones((2,2),dtype=float_),ones((2,2),dtype=float_)


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
    interpol= zeros(nbpoint)
    interpolMask = zeros(nbpoint)
    pointInterpol = zeros((3,nbpoint))
    pointInterpol[0,:] = line(pointImg[0,0],pointImg[0,1],nbpoint)
    pointInterpol[1,:] = line(pointImg[1,0],pointImg[1,1],nbpoint)
   
    
    mask = Slice.get_mask()
    map_coordinates(Slice.get_slice().get_fdata(), pointInterpol , output=interpol, order=1, mode='constant', cval=0, prefilter=False)
    map_coordinates(mask, pointInterpol, output=interpolMask, order=0, mode='constant',cval=0,prefilter=False)
    
    index =~isnan(interpol) * interpolMask>0
    #val_mask = interpol * interpolMask
    #index=val_mask>0
      
    return interpol,index,somme(index)


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
    
    valindex=linspace(0,nbpoint-1,nbpoint,dtype=int)
    if intersection=='union':
        index = index1+index2 
    
    elif intersection=='intersection':
        index = index1*index2
    
    else:
        print(intersection, "is not recognized, choose either 'itersection' or 'union'")
    
    index = valindex[index==True]


    commonVal1 = zeros(val1.shape[0])  
    commonVal1[~isnan(val1)] = val1[~isnan(val1)]
    commonVal2 = zeros(val2.shape[0])  
    commonVal2[~isnan(val2)] = val2[~isnan(val2)] 
 
    return commonVal1[index],commonVal2[index],index
    

@jit(nopython=True,fastmath=True)
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
    return somme((commonVal1 - commonVal2)**2)
    
def ncc(commonVal1,commonVal2):
    
    x=commonVal1[~isnan(commonVal1)]
    y=commonVal2[~isnan(commonVal2)]
    mux = mean(x)
    muy = mean(y)
    stdx = std(x)
    #print(varx)
    stdy = std(y)
    #print(vary)
    res=-1
    if stdx>0 and stdy>0:
        res = (1/(len(x)-1))*somme((x-mux)*(y-muy))/(stdx*stdy)
        
    return res


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
    pt1,pt2,val1,val2,index1,index2,nbpoint,ok = commonSegment(slice1,M1,slice2,M2,res)
    ok=int(ok[0,0]); nbpoint=int(nbpoint[0,0]) #ok and nbpoints are 2-size vectors to allow using numba with this function
    newError=0; commonPoint=0; inter=0; union=0; ncc_var=-1; mse_coupe=(-1,-1)
    #print(ok)
    
    if ok>0:
        #val1,index1,nbpointSlice1=sliceProfil(slice1, pointImg1, nbpoint)  #profile in slice 1
        #print(index1)
        #val2,index2,nbpointSlice2=sliceProfil(slice2, pointImg2, nbpoint)  #profile in slice 2
        #print(index2)
        #commonVal1,commonVal2,index=commonProfil(val1, index1, val2, index2,nbpoint) #union between profile in slice 1 and 2 (used in the calcul of MSE and DICE)
        index=logical_or(index1,index2)
        commonVal1=val1[index]
        commonVal2=val2[index]
        #
        #val1inter,val2inter,interpoint=commonProfil(val1, index1, val2, index2,nbpoint,'intersection') #intersection between profile in slice 1 and 2 (used in the calcul of DICE)
        #numbers of points along union profile (for MSE)
        index=logical_and(index1,index2)
        val1inter=val1[index]
        val2inter=val2[index]
        newError=error(commonVal1,commonVal2)
        if len(val1inter) !=0:
            mse_coupe=(error(val1inter,val2inter),len(val1inter))
        #mse_coupe=0
        commonPoint=len(commonVal1)
        inter=len(val1inter) #number of points along intersection profile (for DICE)
        nbpointSlice1=sum(index1)
        nbpointSlice2=sum(index2)
        union=nbpointSlice1+nbpointSlice2 #number of points along union profile (for DICE)
        #if len(commonVal1) != 0:
            #print(commonVal1,commonVal2)
        #    ncc_var=ncc(commonVal1,commonValFdi2)
        ncc_varr=0
        
    return newError,commonPoint,2*inter,union,ncc_var,mse_coupe

def updateCostBetweenAllImageAndOne(indexSlice,listSlice,gridError,gridNbpoint,gridInter,gridUnion):
    """
    The function update the two matrix used to compute MSE, when one slice position is modified

    Inputs:
    slice1 : slice
        type slice, contains all the necessary information about the slice
    listSlice : list of slices
        contains the slices from all stacks

    """

    slice1=listSlice[indexSlice] #the slice that moves in the lis
    #i_slice1=indexOf(slice1)
    i_slice2=0
    while i_slice2 < indexSlice :
        #i_slice2=where(listSlice==slice2)[0]
        #print(i_slice2)
        slice2=listSlice[i_slice2] #one slice in the list       
        if slice1.get_orientation() != slice2.get_orientation(): #we compute the error between slice1 and slice2 only if they are not from the same stacks
                #if indexSlice > i_slice2 : #The error matrix is triangular as error between slice i and j is the same than between slice j and i. We choose to enter only the values when i>j.
                gridError[indexSlice,i_slice2],gridNbpoint[indexSlice,i_slice2],gridInter[indexSlice,i_slice2],gridUnion[indexSlice,i_slice2],_,_=costLocal(slice1,slice2)               #else:
        i_slice2+=1
    while i_slice2 < len(listSlice):
        slice2=listSlice[i_slice2]
        if slice1.get_orientation() != slice2.get_orientation():            
                gridError[i_slice2,indexSlice],gridNbpoint[i_slice2,indexSlice],gridInter[i_slice2,indexSlice],gridUnion[i_slice2,indexSlice],_,_=costLocal(slice2,slice1)
        i_slice2+=1
    
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
    gridError=zeros((nbSlice,nbSlice))
    gridNbpoint=zeros((nbSlice,nbSlice))
    gridInter=zeros((nbSlice,nbSlice))
    gridUnion=zeros((nbSlice,nbSlice))
    i_slice1=0
    

    for i_slice1 in range(nbSlice): 
        slice1=listSlice[i_slice1]
        if slice1.ok==1 :
            for i_slice2 in range(nbSlice):
                slice2=listSlice[i_slice2]
                if slice2.ok ==1:
                        if (i_slice1 > i_slice2):
                            if slice1.get_orientation() != slice2.get_orientation():
                                newError,commonPoint,inter,union,_,_=costLocal(slice1,slice2) #computed cost informations between two slices
                                gridError[i_slice1,i_slice2]=newError 
                                gridNbpoint[i_slice1,i_slice2]=commonPoint
                                gridInter[i_slice1,i_slice2]=inter
                                gridUnion[i_slice1,i_slice2]=union
                    
    return gridError,gridNbpoint,gridInter,gridUnion     


@jit(nopython=True)
def costFromMatrix(gridNum,gridDenum):
    
    """
    Compute the cost on all slices from two matrix. Can be equally used to compute the MSE and the DICE
    
    """
    
    
    globalNum = somme(gridNum)
    globalDenum= somme(gridDenum)
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
   
    # mean_sum = 0
    # n = 0
    # std=0
    # var = []
    # for s in listSlice:  
    #     data = s.get_slice().get_fdata()*s.get_mask()
    #     X,Y,Z = data.shape
    #     for x in range(X):
    #         for y in range(Y):
    #             if data[x,y,0]> 1e-10:
    #                 mean_sum = mean_sum + data[x,y,0]
    #                 n = n + 1
    #                 var.append(data[x,y,0])
    # mean = mean_sum/n
    # std = sqrt((sum((array(var)-mean)*(array(var)-mean))/(n)))
    # print('avant :',mean,std)
    # std2 = std(var)
    data  = concatenate([s.get_slice().get_fdata().reshape(-1) for s in listSlice])
    mask = concatenate([s.get_mask().reshape(-1) for s in listSlice])
    var = data[mask>0]
    mean = np.mean(var)
    std = np.std(var)
    #print('mean :',mean)
    #print('std :',std)
    #print('maintenant :',mean,std)
    listSliceNorm = []
    for s in listSlice:
        slices = s.get_slice().get_fdata() 
        X,Y,Z = slices.shape
        newSlice = zeros((X,Y,1))
        for x in range(X):
            for y in range(Y):
                newSlice[x,y,0] = (slices[x,y,0] - mean)/std
        newNifti = Nifti1Image(newSlice, s.get_slice().affine)
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
    # std = sqrt((sum((array(var)-mean)*(array(var)-mean))/(n-1))) 
    return listSliceNorm
 

def global_optimisation(hyperparameters,listSlice,ablation):
    
    tps = perf_counter()

    hyperparameters = asarray(hyperparameters,dtype=float_)

    """
    Compute the optimised parameters for each slice. At the end of the function parameters of each slice must be the optimised parameters. The function returns the evolution of the registration on each iterarion.
    
    Input
    
    listSlice : list of sliceObjects
        list of slices from all staks

    Returns
    
    dicRes : A dictionnay wich contains informations on the evolution of the registration during all iteration (ex : evolution of MSE, evolution of DICE, evolution of parameters ....)
    rejectedSlice : list of rejected slices and their corresponding stack
    
    """
    print('ablattion 1 :',ablation[0])
    print('ablation 2 :',ablation[1] )
    #Initialisation
    nbSlice=len(listSlice) 
    EvolutionError=[] 
    EvolutionDice=[]
    EvolutionGridError=[] 
    EvolutionGridNbpoint=[]
    EvolutionGridInter=[]
    EvolutionGridUnion=[]
    EvolutionParameters=[] 
    Previous_parameters=zeros((6,nbSlice))
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
    set_o=zeros(nbSlice)
    

    grid_slices=array([gridError,gridNbpoint,gridInter,gridUnion])
    set_r=zeros(nbSlice)
  
    if ablation[0]=='no_gaussian':
         hyperparameters[5] = 0
    elif ablation[0]=='no_dice':
         hyperparameters[4] = 0
    #hyperparameters[4]=0 #no dice
    #hyperparameters[5]=0 #no gaussian filtering -> we find out it have no interest in the algorithm
    # # print('ablation :', ablation)
    # # print('hyperparameters :', hyperparameters)
    nbstack=max([slicei.get_orientation() for slicei in listSlice])
    #print(nbstack)
    res=zeros(nbstack+1)
    nbslice=len(listSlice)
    for nb in range(0,nbstack+1):
        index_image=where([listSlice[i_slice].get_orientation()==nbstack for i_slice in linspace(0,nbslice-1,nbslice,dtype=int)])
        #print(index_image)
        image=array(listSlice)[index_image]
        #print(image)
        res[nb]=max(image[0].get_slice().header.get_zooms())
        print('res :',res)
    
    len(res)

    Vmx=computeMaxVolume(listSlice,nbstack,res)
    #if hyperparameters[5] != 0 :
    print('--------------motion estimation with gaussian filtering and dice-----------------')
    print('---First Iteration -----')
    tpsg1 =perf_counter()
    print('Hyperparameters :')
    print('Initial Simplex :',hyperparameters[0])
    print('Final Simplex :',hyperparameters[1])
    print('Epsilon :',sqrt(6*hyperparameters[3]**2))
    print('Dice weight :',hyperparameters[4])
    print('Gaussian blurring :',hyperparameters[5])
    new_hyperparameters = array([hyperparameters[0],hyperparameters[1],hyperparameters[2],sqrt(6*hyperparameters[3]**2),hyperparameters[4],hyperparameters[5]])
    ge,gn,gi,gu,dicRes=algo_optimisation(new_hyperparameters,listSlice,set_o,set_r,grid_slices,dicRes,Vmx,10)
    grid_slices=array([ge,gn,gi,gu])
    set_r=zeros(nbSlice)
    #     #Second Step : sigma = 2.0, d=b/2, x_opt, omega
    #     #epsilon = sqrt(6*(erreur/2)^2)
    tpeg1 = perf_counter()
    tp = tpeg1 - tpsg1
    hours = int(tp//(60*60))
    minutes = int((tp%(60*60))//60)
    secondes = int(tp%60)
    print('Fist Iteration was done in %s hours %s minutes and %s seconds' %(hours,minutes,secondes))
       
    print('----Second Iteration----')
    tpsg2 = perf_counter()
    print('Hyperparameters :')
    print('Initial Simplex :',hyperparameters[0]/2)
    print('Final Simplex :',hyperparameters[1]/2)
    print('Epsilon :',sqrt(6*(hyperparameters[3]/2)**2))
    print('Dice Weight :',hyperparameters[4])
    print('Gaussien blurring :',hyperparameters[5]/2)
    new_hyperparameters = array([hyperparameters[0]/2,hyperparameters[1]/2,hyperparameters[2],sqrt(6*(hyperparameters[3]/2)**2),hyperparameters[4],hyperparameters[5]/2])
    ge,gn,gi,gu,dicRes=algo_optimisation(new_hyperparameters,listSlice,set_o,set_r,grid_slices,dicRes,Vmx,10)
    grid_slices=array([ge,gn,gi,gu])
    tpeg2 = perf_counter()
    tp = tpeg2 - tpsg2
    hours = int(tp//(60*60))
    minutes = int((tp%(60*60))//60)
    secondes = int(tp%60)
    print('Second Iteration was done in %s hours %s minutes and %s seconds' %(hours,minutes,secondes))
    # #Third step : sigma = 0.0, d=b/4, x_opt, omega
    # #epsilon = sqrt(6*(erreur)^2/4)
    #if hyperparameters[4] != 0 :
    print('---motion estimation without gaussian filtering but with dice----')
    tpsd = perf_counter()
    print('Hyperparameters :')
    print('Initial Simplex :',hyperparameters[0]/4)
    print('FInal Simplex :',hyperparameters[1]/4)
    print('Epsilon :',sqrt(6*(hyperparameters[3]/4)**2))
    print('Dice Weight :',hyperparameters[4])
    print('Gaussien blurirng :',0)
    new_hyperparameters = array([hyperparameters[0]/4,hyperparameters[1]/4,hyperparameters[2],sqrt(6*(hyperparameters[3]/4)**2),hyperparameters[4],0])
    ge,gn,gi,gu,dicRes=algo_optimisation(new_hyperparameters,listSlice,set_o,set_r,grid_slices,dicRes,Vmx,10)
    grid_slices=array([ge,gn,gi,gu])
    tped = perf_counter()
    tp = tped - tpsd
    hours = int(tp//(60*60))
    minutes = int((tp%(60*60))//60)
    secondes = int(tp%60)
    print('Third Iteration was done in %s hours %s minutes and %s seconds' %(hours,minutes,secondes))
    # #Fourth step : sigma = 0.0 , d=b/8, x_opt, omega
    # #epsilon = sqrt(6*(erreur)^2/8)
    print('---motion estmisation without gaussien filtering and without dice----')
    tpsnod = perf_counter()
    print('Hyperparameters :')
    print('Initial simplex :',hyperparameters[0]/8)
    print('Final simplex :',hyperparameters[1]/8)
    print('Epsilon :',sqrt(6*(hyperparameters[3]/8)**2))
    print('Dice Weight :',0)
    print('Gaussien blurring :', 0)
    if ablation[1]=="no_dice":
        new_hyperparameters = array([hyperparameters[0]/8,hyperparameters[1]/8,hyperparameters[2],sqrt(6*(hyperparameters[3]/8)**2),hyperparameters[4],0])
    else :
        new_hyperparameters = array([hyperparameters[0]/8,hyperparameters[1]/8,hyperparameters[2],sqrt(6*(hyperparameters[3]/8)**2),0,0])
    ge,gn,gi,gu,dicRes=algo_optimisation(new_hyperparameters,listSlice,set_o,set_r,grid_slices,dicRes,Vmx,10)
    grid_slices=array([ge,gn,gi,gu])
    tpenod = perf_counter()
    tp = tpenod - tpsnod
    hours = int(tp//(60*60))
    minutes = int((tp%(60*60))//60)
    secondes = int(tp%60)
    print('Fourth Iteration was done in %s hours %s minutes and %s seconds' %(hours,minutes,secondes))
    
    ge=grid_slices[0,:,:]
    gn=grid_slices[1,:,:]
    gi=grid_slices[2,:,:]
    gu=grid_slices[3,:,:]
    
    #union=zeros((nbSlice,nbSlice))
    #intersection=zeros((nbSlice,nbSlice))
    nbSlice=len(listSlice)
    #union=zeros((nbSlice,nbSlice))
    #intersection=zeros((nbSlice,nbSlice))

    #set_o = zeros(nbSlice)
    #print("badly register : ",sum(set_o))
    #print(set_o)
    #ge,gn,gi,gu,dicRes=algo_optimisation(new_hyperparameters,listSlice,set_o,set_o,grid_slices,dicRes)  
    #grid_slices=array([ge,gn,gi,gu])
      
    if ablation[0]!='no_multistart':
    #set_o1,set_o2 = detect_misregistered_slice(listSlice, grid_slices, loaded_model)
    #set_r = logical_or(set_o1,set_o2) 
    #print(set_o)
        #before = removeBadSlice(listSlice,set_o)
        new_hyperparameters = array([hyperparameters[0],hyperparameters[1],hyperparameters[2],sqrt(6*hyperparameters[3]**2),hyperparameters[4],0])
        #grid_slices,set_r,dicRes=correction_out_images(listSlice,new_hyperparameters,set_o1.copy(),set_r,grid_slices,dicRes)
        print('---------------Outliers\'detection -------')
        set_o1,set_o2 = detect_misregistered_slice(listSlice, grid_slices, loaded_model)
        set_o = np.logical_or(set_o1,set_o2)
        ge,gn,gi,gu,dicRes=algo_optimisation(new_hyperparameters,listSlice,set_o,set_o,grid_slices,dicRes,Vmx,10)
    #set_r = logical_or(set_o1,set_o2) 
   
        #grid_slices,set_r,dicRes=correction_out_images(listSlice,new_hyperparameters,set_o.copy(),set_o2,grid_slices,dicRes)
        #set_o1,set_o2 = detect_misregistered_slice(listSlice, grid_slices, loaded_model)
        #set_o = np.logical_or(set_o1,set_o2)
        grid_slices,set_r,dicRes=correction_misregistered(listSlice,new_hyperparameters,set_o.copy(),set_o.copy(),grid_slices,dicRes,Vmx)
        ge,gn,gi,gu,dicRes=algo_optimisation(new_hyperparameters,listSlice,set_r,set_r,grid_slices,dicRes,Vmx,10)
        rejectedSlices=removeBadSlice(listSlice, set_r)
        
    else:
        rejectedSlices=[]
    #rejectedSlices=[]
    tpe = perf_counter()
    tp = tpe-tps
    hours = int(tp//(60*60))
    minutes = int((tp%(60*60))//60)
    secondes = int(tp%60)
    print('The global optimisation was done in %s hours %s minutes and %s seconds' %(hours,minutes,secondes))
    
    return dicRes, rejectedSlices
    
    

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


def SliceOptimisation1Image(d,xatol,listSlice,gridError,gridNbpoint,gridInter,gridUnion,initial_s,finish,lamb,nImage,Vmx):    
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
        x_opt=SimplexOptimisation(delta,xatol, x0, nImage, listSlice, gridError, gridNbpoint, gridInter, gridUnion, initial_s, lamb,Vmx)
        current_param=x_opt
        
        if norm(previous_param-current_param)<1e-1:
            finish_val=1
    else:
        x_opt=x0

    return x_opt,finish_val          

def oneSliceCost(grid_numerator,grid_denumerator,set_o,i_slice):
    """
    Compute the cost of one slice, using the two cost matrix. Can be used equally for MSE and DICE
    """
    set_outliers = abs(1-set_o.copy())
    set_outliers[i_slice]=1
    nbSlice=shape(grid_numerator)[0]
    
    grid_numerator_no_o=[sqrt(x*y) for x,y in zip(grid_numerator*set_outliers,transpose(transpose(grid_numerator)*set_outliers))]
    grid_denumerator_no_o=[sqrt(x*y) for x,y in zip(grid_denumerator*set_outliers,transpose(transpose(grid_denumerator)*set_outliers))]
     
    numerator = somme(grid_numerator_no_o)

    num=array(grid_numerator_no_o)
    num=reshape(num,(nbSlice,nbSlice))
    num=somme(num[i_slice,:])+somme(num[:,i_slice])
    denum=array(grid_denumerator_no_o)
    denum=reshape(denum,(nbSlice,nbSlice))
    denum=somme(denum[i_slice,:])+somme(denum[:,i_slice])
    
    
    if denum==0:
        cost=nan
    else:
        cost=num/denum
    if isinf(cost):
        print('ALERT ERROR')
    return cost



def correction_out_images(listSlice,hyperparameters,set_o,set_r,grid_slices,dicRes,Vmx):
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
    print(set_o.shape)
    ge = grid_slices[0,:,:]
    gn = grid_slices[1,:,:]
    gi = grid_slices[2,:,:]
    gu= grid_slices[3,:,:]
    before_correction = sum(set_o)
    print(before_correction)
     
    nbSlice = len(listSlice)     
    while 1:
                
        #number of slices total
        print(set_o.shape)
        for i_slice in range(0,nbSlice):
            
        #     #print('before:',listSlice[i_slice].get_parameters())
        #     #print('Slice Cost Before:', oneSliceCost(ge,gn,set_o,i_slice))
            
            n_0=somme(gn[:,i_slice])+somme(gn[i_slice,:])
            okpre=False;okpost=False;nbSlice1=0;nbSlice2=0;dist1=0;dist2=0
            slicei=listSlice[i_slice]
            x_pre=slicei.get_parameters()
            print(set_o[i_slice])
            if set_o[i_slice]==1:
                it=0
                while it<5:
                    print('i_slice',i_slice,'it',it) 
                    for i in range(i_slice-2,0,-2):
                        if listSlice[i].get_orientation()==listSlice[i_slice].get_orientation(): #check if the previous slice is from the same volume
                            if set_r[i]==0 : #if the previous slice is well registered 
                                nbSlice1=i
                                dist1=abs(i_slice-nbSlice1)//2
                                okpre=True
                                break
                    for j in range(i_slice+2,nbSlice,2):
                        if set_r[j]==0: #if the previous slice is well registered
                            if listSlice[j].get_orientation()==listSlice[i_slice].get_orientation(): 
                                nbSlice2=j
                                dist2=abs(i_slice-nbSlice2)//2
                                okpost=True
                                break
                    if okpre==True and okpost==True: #if there is two close slice well-register, we do a mean between them
                        Slice1=listSlice[nbSlice1];Slice2=listSlice[nbSlice2]
                        ps1=Slice1.get_parameters().copy();ps2=Slice2.get_parameters().copy();
                        print('ps1:',ps1,'dist1:',dist1)
                        print('ps2:',ps2,'dist2:',dist2)
                        #MS1=rigidMatrix(ps1);MS2=rigidMatrix(ps2)
                        MS1=Slice1.get_transfo();MS2=Slice2.get_transfo()
                        RotS1=MS1[0:3,0:3];RotS2=MS2[0:3,0:3]
                        TransS1=MS1[0:3,3];TransS2=MS2[0:3,3]
                        Rot=computeMeanRotation(RotS1,dist1,RotS2,dist2)
                        Trans=computeMeanTranslation(TransS1,dist1,TransS2,dist2)
                        Mtot=eye(4)
                        Mtot[0:3,0:3]=Rot;Mtot[0:3,3]=Trans
                        #x0=ParametersFromRigidMatrix(Mtot)
                        center = Slice1.get_center()
                        center_mat = eye(4)
                        center_mat[0:3,3] = center
                        center_inv = eye(4)
                        center_inv[0:3,3] = -center
                        M_est = center_mat @ Mtot @ inv(Slice1.get_slice().affine) @ center_inv
                        x0 = ParametersFromRigidMatrix(M_est)
                        print('x0',x0)
                            #slicei.set_parameters(x0)
                            
                    elif okpre==True and okpost==False: #if there is only one close slice well-register, we take the parameters of this slice
                        Slice1=listSlice[nbSlice1]
                        x0=Slice1.get_parameters().copy()
                        print('x0',x0)
                        #slicei.set_parameters(p)
                    elif okpost==True and okpre==False: 
                        Slice2=listSlice[nbSlice2]
                        x0=Slice2.get_parameters().copy()
                        print('x0',x0)
                        #slicei.set_parameters(p)
                    else :
                        x0=slicei.get_parameters()
                        print('x0',x0)
                        break
                            
                #         print(i_slice,slicei.get_orientation(),slicei.get_index_slice(),okpre,okpost,x0)
            
            
                    #We do a multistart optimisation based on the new initialisation
                    #print('x0:',x0) 
                    multistart =zeros((6,6))
                    multistart[:5,:]=(line(-20,20,5)*ones((6,5))).T
                    #print(multistart)
                    index = array([0,1,2,3,4,5])
                    with Pool(processes=16) as p:
                        tmpfun=partial(multi_start,hyperparameters,i_slice,listSlice.copy(),grid_slices,set_o,x0,multistart,Vmx)
                        res=p.map(tmpfun,index)
                            
                        #
                    #cost=opti_res[0]
                    #x_opt=opti_res[3]
                    current_cost=cost_from_matrix(ge,gn,set_o,i_slice)
                    #p=array([p[0] for p in res])
                        
                    #print(p)
                    #p.append(cost)
                        
                    x=res
                    n = zeros(len(x))
                    p = zeros(len(x))
                    i_n=0
                    for x_tmp in x:
                        listSlice[i_slice].set_parameters(x_tmp)
                        getmp=ge.copy()
                        gntmp=gn.copy()
                        gitmp=gi.copy()
                        gutmp=gu.copy()
                        updateCostBetweenAllImageAndOne(i_slice,listSlice,getmp,gntmp,gitmp,gutmp)
                        grid_tmp=array([getmp,gntmp,gitmp,gutmp])
                        p[i_n]=cost_fct(x_tmp,i_slice,listSlice,grid_tmp,set_o,1,Vmx)
                        n[i_n]=somme(gitmp[i_slice,:]+somme(gitmp[:,i_slice]))
                        i_n=i_n+1
                        
                        
                    good_index = where(n>n_0)[0]
                    print('n_0',n_0,n[good_index])

                    p = p[good_index]
                    #si p est vide, on relance avec un simplex plus petit, sinon on garde les même valeurs.
                    if len(p)==0:
                        delta = hyperparameters[0]
                        d = delta/2
                        hyperparameters[0]=d
                        it=it+1
                        print('je suis ici')
                    else :
                        print('je suis la')
                        p = p.tolist()
                        print(x)
                        x=array([x])[0,:,:]
                        print(x.shape)
                        print(good_index)
                        x = x[good_index]
                        x = x.tolist()
                        n = n[good_index]
                        n = n.tolist()
                        mincost = min(p)
                        x_opt = x[p.index(mincost)]
                        n_1 = n[p.index(mincost)]
                        print('new_n',n_1)
                        print('corrected x',x_opt)
                        #print('i_slice',i_slice)
                        print('After:', x_opt)
                        print('mincost :',mincost)
                        x_opt = np.array([x_opt])
                        print(x_opt)
                        slicei.set_parameters(x_opt[0,:])
                            
                            
                        updateCostBetweenAllImageAndOne(i_slice,listSlice,ge,gn,gi,gu)
                            
                        grid_slices = array([ge,gn,gi,gu])
                        #print('Slice Cost After:',i_slice, oneSliceCost(ge,gn,set_o,i_slice))
                        costMse=costFromMatrix(ge,gn)
                        #print('cost :', costMse)
                        costDice=costFromMatrix(gi,gu)
                        
                        ge,gn,gi,gu,dicRes=algo_optimisation(hyperparameters,listSlice,set_r,set_o,grid_slices,dicRes,Vmx,10)  
                        grid_slices=array([ge,gn,gi,gu]) 
                        
                        new_set_o1,new_set_o2 = detect_misregistered_slice(listSlice,grid_slices,loaded_model)
                        print('La coupe est-elle corrigé ?',i_slice,new_set_o2[i_slice])
                        #set_o = logical_or(new_set_o1,new_set_o2)
                            
                        
                        break 

        #intersection=zeros((nbSlice,nbSlice))
        #union=zeros((nbSlice,nbSlice))
        #mW = matrixOfWeight(ge, gn,gi,gu, intersection, union, listSlice,t_inter,t_mse, t_inter, t_mse)
        ge,gn,gi,gu = computeCostBetweenAll2Dimages(listSlice)
        print(costFromMatrix(ge,gn))
        #grid_slices = array([ge,gn,gi,gu])
        #ge,gn,gi,gu,dicRes=algo_optimisation(hyperparameters,listSlice,set_o,set_o,grid_slices,dicRes)  
        
        #print("badly register : ", sum(new_set_o))
    
        new_set_o1,new_set_o2 = detect_misregistered_slice(listSlice,grid_slices,loaded_model)
        set_o = logical_or(new_set_o1,new_set_o2)
        print('mis-registered-before :', removeBadSlice(listSlice,set_o))
        print('mis-registered-after :', removeBadSlice(listSlice,new_set_o2))
        
        if all(new_set_o2 == set_r):
            break
            
        set_r = new_set_o2.copy()
    
    set_r = new_set_o2
    after_correction = sum(set_r)
    saved = before_correction - after_correction
    #print('slices saved with multistart :', saved)
    ge,gn,gi,gu=computeCostBetweenAll2Dimages(listSlice)
    costMse=costFromMatrix(ge,gn)
    costDice=costFromMatrix(gi,gu)
    updateResults(dicRes, ge, gn, gi, gu, costMse, costDice, listSlice, nbSlice)
    grid_slices = array([ge,gn,gi,gu])
        
    return grid_slices,set_o,dicRes

##deux fonctions pour voir si ça marche mais après il faudra factoriser
def correction_misregistered(listSlice,hyperparameters,set_o,set_r,grid_slices,dicRes,Vmx):
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
    print(set_o.shape)
    ge = grid_slices[0,:,:]
    gn = grid_slices[1,:,:]
    gi = grid_slices[2,:,:]
    gu = grid_slices[3,:,:]
    before_correction = sum(set_o)
    print(before_correction)
     
    nbSlice = len(listSlice)     
    while True:
        
                
         #number of slices total
        print(set_o.shape)
        for i_slice in range(0,nbSlice):
            
        #     #print('before:',listSlice[i_slice].get_parameters())
        #     #print('Slice Cost Before:', oneSliceCost(ge,gn,set_o,i_slice))
             n_0=somme(gn[:,i_slice])+somme(gn[i_slice,:])
             okpre=False;okpost=False;nbSlice1=0;nbSlice2=0;dist1=0;dist2=0
             slicei=listSlice[i_slice]
             x_pre=slicei.get_parameters()
             print(set_o[i_slice])
             if set_o[i_slice]==1:
                 #print(i_slice)
                 for i in range(i_slice-2,0,-2):
                     if listSlice[i].get_orientation()==listSlice[i_slice].get_orientation(): #check if the previous slice is from the same volume
                         if set_r[i]==0 : #if the previous slice is well registered 
                             nbSlice1=i
                             dist1=abs(i_slice-nbSlice1)//2
                             okpre=True
                             break
                 for j in range(i_slice+2,nbSlice,2):
                     if set_r[j]==0: #if the previous slice is well registered
                         if listSlice[j].get_orientation()==listSlice[i_slice].get_orientation(): 
                              nbSlice2=j
                              dist2=abs(i_slice-nbSlice2)//2
                              okpost=True
                              break
                 if okpre==True and okpost==True: #if there is two close slice well-register, we do a mean between them
                      Slice1=listSlice[nbSlice1];Slice2=listSlice[nbSlice2]
                      ps1=Slice1.get_parameters().copy();ps2=Slice2.get_parameters().copy();
                      print('ps1:',ps1,'dist1:',dist1)
                      print('ps2:',ps2,'dist2:',dist2)
                      #MS1=rigidMatrix(ps1);MS2=rigidMatrix(ps2)
                      MS1=Slice1.get_transfo();MS2=Slice2.get_transfo()
                      RotS1=MS1[0:3,0:3];RotS2=MS2[0:3,0:3]
                      TransS1=MS1[0:3,3];TransS2=MS2[0:3,3]
                      Rot=computeMeanRotation(RotS1,dist1,RotS2,dist2)
                      Trans=computeMeanTranslation(TransS1,dist1,TransS2,dist2)
                      Mtot=eye(4)
                      Mtot[0:3,0:3]=Rot;Mtot[0:3,3]=Trans
                      #x0=ParametersFromRigidMatrix(Mtot)
                      center = Slice1.get_center()
                      center_mat = eye(4)
                      center_mat[0:3,3] = center
                      center_inv = eye(4)
                      center_inv[0:3,3] = -center
                      M_est = center_mat @ Mtot @ inv(Slice1.get_slice().affine) @ center_inv
                      x0 = ParametersFromRigidMatrix(M_est)
                      print('x0',x0)
                      #slicei.set_parameters(x0)
                      
                 elif okpre==True and okpost==False: #if there is only one close slice well-register, we take the parameters of this slice
                      Slice1=listSlice[nbSlice1]
                      x0=Slice1.get_parameters().copy()
                      print('x0',x0)
                      #slicei.set_parameters(p)
                 elif okpost==True and okpre==False: 
                      Slice2=listSlice[nbSlice2]
                      x0=Slice2.get_parameters().copy()
                      print('x0',x0)
                      #slicei.set_parameters(p)
                 else :
                      x0=slicei.get_parameters()
                      print('x0',x0)
                     
        #         print(i_slice,slicei.get_orientation(),slicei.get_index_slice(),okpre,okpost,x0)
    
    
                #We do a multistart optimisation based on the new initialisation
                 #print('x0:',x0) 
                 multistart = zeros((6,6))
                 multistart[:5,:]=(line(-20,20,5)*ones((6,5))).T
                 #print(multistart)
                 index = array([0,1,2,3,4,5])
                 with Pool(processes=16) as p:
                     tmpfun=partial(multi_start,hyperparameters,i_slice,listSlice.copy(),grid_slices,set_o,x0,multistart,Vmx)
                     res=p.map(tmpfun,index)
                 print(res)
                #
                #cost=opti_res[0]
                #x_opt=opti_res[3]
                 current_cost=cost_from_matrix(ge,gn,set_o,i_slice)
                 #p=array([p[0] for p in res])
                 
                 #print(p)
                 #p.append(cost)
                 
                 x=res#array([x[0] for x in res]) 
                 n = zeros(len(x))
                 p = zeros(len(x))
                 i_n=0
                 for x_tmp in x:
                     print(x_tmp)
                     listSlice[i_slice].set_parameters(x_tmp)
                     ge_tmp=ge.copy()
                     gn_tmp=gn.copy()
                     gi_tmp=gi.copy()
                     gu_tmp=gu.copy()
                     updateCostBetweenAllImageAndOne(i_slice,listSlice,ge_tmp,gn_tmp,gi_tmp,gu_tmp)
                     grid_tmp = array([ge_tmp,gn_tmp,gi_tmp,gu_tmp])
                     p[i_n]=cost_fct(x_tmp,i_slice,listSlice,grid_tmp,set_o,1,Vmx)
                     n[i_n]=somme(gn[i_slice,:]+somme(gn[:,i_slice]))
                     i_n=i_n+1
                 
            
                 mincost = min(p)
                 p=p.tolist()
                 x=x
                 x_opt = x[p.index(mincost)]
                 print('corrected x',x_opt)
                 #print('i_slice',i_slice)
                 print('After:', x_opt)
                 print('mincost :',mincost)
                 slicei.set_parameters(x_opt)
                
                
                 updateCostBetweenAllImageAndOne(i_slice,listSlice,ge,gn,gi,gu)
                
                
                 grid_slices = array([ge,gn,gi,gu])
                 #print('Slice Cost After:',i_slice, oneSliceCost(ge,gn,set_o,i_slice))
                 costMse=costFromMatrix(ge,gn)
                 #print('cost :', costMse)
                 costDice=costFromMatrix(gi,gu)
                 
                 new_set_o1,new_set_o2 = detect_misregistered_slice(listSlice,grid_slices,loaded_model)
                 set_o = logical_or(new_set_o1,new_set_o2) 
                 
                 hyperparameters[0]=4
                 ge,gn,gi,gu,dicRes=algo_optimisation(hyperparameters,listSlice,set_o,set_o,grid_slices,dicRes,Vmx,10)  
                 grid_slices=array([ge,gn,gi,gu])
                 
        
        #intersection=zeros((nbSlice,nbSlice))
        #union=zeros((nbSlice,nbSlice))
        #mW = matrixOfWeight(ge, gn,gi,gu, intersection, union, listSlice,t_inter,t_mse, t_inter, t_mse)
        ge,gn,gi,gu = computeCostBetweenAll2Dimages(listSlice)
        print(costFromMatrix(ge,gn))
        #grid_slices = array([ge,gn,gi,gu])
        ge,gn,gi,gu,dicRes=algo_optimisation(hyperparameters,listSlice,set_o,set_o,grid_slices,dicRes,Vmx,10)  
        
        #print("badly register : ", sum(new_set_o))
    
        new_set_o1,new_set_o2 = detect_misregistered_slice(listSlice,grid_slices,loaded_model)
        set_r = logical_or(new_set_o1,new_set_o2)
        print('mis-registered-before :', removeBadSlice(listSlice,set_o))
        print('mis-registered-after :', removeBadSlice(listSlice,set_r))

        
        
        if all(set_r == set_o):
            break
            
        set_o = set_r.copy()
    
    
    new_set_o1,new_set_o2 = detect_misregistered_slice(listSlice,grid_slices,loaded_model)
    set_o = logical_or(new_set_o1,new_set_o2)
    after_correction = sum(set_o)
    #set_o = new_set_o1
    saved = before_correction - after_correction
    #print('slices saved with multistart :', saved)
    ge,gn,gi,gu=computeCostBetweenAll2Dimages(listSlice)
    costMse=costFromMatrix(ge,gn)
    costDice=costFromMatrix(gi,gu)
    updateResults(dicRes, ge, gn, gi, gu, costMse, costDice, listSlice, nbSlice)
    grid_slices = array([ge,gn,gi,gu])
        
    return grid_slices,set_o,dicRes



def multi_start(hyperparameters,i_slice,listSlice,grid_slices,set_o,x0,valstart,Vmx,index):
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
    x=x0+valstart[index,:]
    print(listSlice[i_slice].get_index_slice(),listSlice[i_slice].get_orientation())
    #print('index multistart:',valstart[index,:],'index:',index)
    x_opt = SimplexOptimisation(x,hyperparameters,listSlice,grid_slices,set_o,i_slice,Vmx)
    #cost=opti_res[0] #-1*opti_res[1]
    #x_opt=opti_res[3]

    
    return x_opt



def removeBadSlice(listSlice,set_o):
    """
    return a list of bad-registered slices and their corresponding stack
    """
    removelist=[]

   
    for i_slice in range(len(listSlice)):
      if set_o[i_slice]==1:
          removelist.append((listSlice[i_slice].get_orientation(),listSlice[i_slice].get_index_slice()))
   
    
    return removelist
        

@jit(nopython=True)
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
    
    nbslice,nbslice = shape(grid_numerator)
    
    grid_numerator_no_o = grid_numerator.copy()
    grid_denumerator_no_o = grid_denumerator.copy()
    
    set_outliers = 1-set_o
    
    set_outliers[i_slice]=1

    #grid_numerator_no_o=array([x for x,y in zip(grid_numerator*set_outliers,transpose(transpose(grid_numerator)*set_outliers))])
    
    #grid_denumerator_no_o=array([sqrt(x*y) for x,y in zip(grid_denumerator*set_outliers,transpose(transpose(grid_denumerator)*set_outliers))])
     

    #numerator = sum(grid_numerator[:,i_slice]*set_outliers)+sum(transpose(transpose(grid_numerator[i_slice,:])*set_outliers))
    #denumerator = sum(grid_denumerator[:,i_slice]*set_outliers)+sum(transpose(transpose(grid_denumerator[i_slice,:])*set_outliers)) #sum(grid_denumerator_no_o)
    #grid_outliers = set_outliers*np.tri(nbslice)
    grid_outliers=zeros((nbslice,nbslice))
    #print((nbslice*nbslice)/2,sum(grid_outliers))
    
    #for slice1 in range(0,nbslice):
    #    for slice2 in range(0,nbslice):
    #        if slice1>slice2:
    #            grid_outliers[slice1,slice2]=set_outliers[slice1]*set_outliers[slice2]
   
    numerator = sum(grid_numerator_no_o)# * grid_outliers)
    denumerator = sum(grid_denumerator_no_o)# * grid_outliers)
    #print(grid_numerator_no_o.shape)
    #cost = costFromMatrix(grid_numerator_no_o, grid_denumerator_no_o)
    if denumerator==0:
        cost=nan
    cost=numerator/denumerator
    return cost
            

def cost_fct(x0,i_slice,listSlice,grid_slices,set_o,lamb,Vmx):
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
    
    x = copy(x0) #copy to use bound in miminization fonction
    #print(x)
    slicei = listSlice[i_slice]
    slicei.set_parameters(x)
    #print(x)
    
    updateCostBetweenAllImageAndOne(i_slice,listSlice,ge,gn,gi,gu)
    

    grid_slices = array([ge,gn,gi,gu])
   
    #print(type(set_o[0]))
    #set_o = zeros((len(listSlice)))
    mse = cost_from_matrix(ge,gn,set_o,i_slice)

    nbslice = shape(gi)[0]
    i_slice1 = linspace(0,nbslice,nbslice,dtype=int)
    i_slice2 = linspace(0,nbslice,nbslice,dtype=int)
    index=np.meshgrid(i_slice1,i_slice2)
    bool_ind=index[0]<index[1]
    dice = sum(gi[bool_ind])
    dice=dice/Vmx
    #print(dice)
    #dice = cost_from_matrix(gi,gu,set_o,i_slice)
    #print('nbslice',i_slice,'mse',mse,'dice',dice)
    #print(dice/Vmx)

    cost = mse - lamb*(dice)
    
    return cost


   

def cost_optimisation_step(hyperparameters,listSlice,grid_slices,set_r,set_o,Vmx,n_image):
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
    lamb=hyperparameters[4]
    
    slicei=listSlice[n_image]
    x0=slicei.get_parameters()
    previous_param=x0
    
    registration_state = set_r[n_image]
 
    
    #if registration_state == 0:
    x_opt=SimplexOptimisation(x0,hyperparameters,listSlice,grid_slices,set_o,n_image,Vmx)
    current_param=x_opt
    #print('previous',previous_param,'current',current_param,epsilon)
    if norm(previous_param-current_param)<epsilon:
        registration_state=1
        #print(n_image,'slice ok')
    #else:
    #    x_opt=x0
    
    #print('CostMse',n_image,costMse-lamb*costDice)
    #print('slice',n_image)
    return x_opt,registration_state      
    
def algo_optimisation(hyperparameters,listSlice,set_o,set_r,grid_slices,dicRes,Vmx,max_extern):
    
    """
    function to optimise cost on all slices
    
    hyperparameters: parameters for optimisation (simplex initial size, xatol, fatol, epsilon, gauss, lamb)
        
    listSlice: set of slice
        
    set_o: set of outliers slices
        
    dicRes: save results for representation
    """
    
    
    max_intern = 10
    
    it_extern = 0
    it_intern = 0
    
    images,mask=createVolumesFromAlist(listSlice)
                                       
    nb_images=len(images)         
    nbSlice = len(listSlice)    
    
    blur = hyperparameters[5]
    #print('blur :', blur)
    #print('Hehehe')
    listSliceblur = denoising(listSlice.copy(),blur)
    ge,gn,gi,gu=computeCostBetweenAll2Dimages(listSliceblur)
    grid_slices=array([ge,gn,gi,gu])
    globalConvergence=False

    while it_extern<max_extern and not(globalConvergence): #(it_intern > 1 or it_extern == 0) and it_extern < max_extern:
        globalConvergence=True
        print('Number of iteration : ', it_extern)
        #print("it_extern:",it_extern)
        #print("it_intern:",it_intern)
        
        set_to_register=abs(set_r.copy()) #at the begenning, we need to register all the slices
        #print(set_to_register)
        set_to_register_pre = zeros(nbSlice)
        it_intern = 0
        while set_to_register.all()!=1 and it_intern < max_intern :#and (not sum(set_to_register)==sum(set_to_register_pre) or sum(set_to_register_pre)==0) :
            #avoid that the algorithm continue because there is one or two slices not register
            #set_to_register_pre = set_to_register.copy()
            #index_pre=0
            #for n_image in range(nb_images):
                #start = )
                #nbSliceImageN=len(images[n_image])
                #randomIndex= arange(nbSliceImageN) 
                #index=randomIndex+index_pre
                #eval_index=where(set_to_register[index]==0)[0]
                #eval_index=eval_index+index_pre
            eval_index=linspace(0,nbSlice-1,nbSlice,dtype=int)
            shuffle(eval_index)
                
                        
            #with Pool(1) as p: #all slices from the same stacks are register together (in the limit of the number of CPU on your machine)
            for i_slice in eval_index:
                if not(set_to_register[i_slice]):
                    #print(i_slice)
                    res=cost_optimisation_step(hyperparameters,listSliceblur,grid_slices,set_to_register,set_o,Vmx,i_slice) 
                                            
                    #for m,i_slice in enumerate(eval_index): #update parameters once the registration is done
                        #print(m,i_slice)
                    listSliceblur[i_slice].set_parameters(res[0])
                    set_to_register[i_slice]=res[1]
                    if not(res[1]):
                        globalConvergence=False
                    updateCostBetweenAllImageAndOne(i_slice,listSliceblur,ge,gn,gi,gu)
                
                
            #ge,gn,gi,gu=computeCostBetweenAll2Dimages(listSliceblur)
            #grid_slices = array([ge,gn,gi,gu])
            ge=grid_slices[0,:,:]
            gn=grid_slices[1,:,:]
            gi=grid_slices[2,:,:]
            gu=grid_slices[3,:,:]
            costMse=costFromMatrix(ge,gn)
            print('costMse:', costMse)
            #costDice=costFromMatrix(gi,gu)
            print('Dice:',somme(gi)/Vmx)
            #index_pre=index_pre+len(images[n_image])
            
            it_intern+=1
            #print('well registered :',sum(set_to_register>0))
            
        

        #print('delta :',  hyperparameters[0], 'iteration :', it_extern, 'final MSE :', costMse, 'final DICE :', costDice)
        it_extern+=1 
        #end = )
        #elapsed = end - start
        
        for i_slice in range(nbSlice):
            listSlice[i_slice].set_parameters(listSliceblur[i_slice].get_parameters())
                        
        ge,gn,gi,gu=computeCostBetweenAll2Dimages(listSlice)
        costMse=costFromMatrix(ge, gn)
        #print('costMse_After',costMse)
        costDice=costFromMatrix(gi,gu)
        updateResults(dicRes,ge,gn,gi,gu,costMse,costDice,listSlice,nbSlice)
        #print('MSE: ', costMse)
        #print('Dice: ', costDice) 
        #print(f'Temps d\'exécution : {elapsed}') 
        
        #it_extern+=1
    
    return ge,gn,gi,gu,dicRes

def SimplexOptimisation(x0,hyperparameters,listSlice,grid_slices,set_o,i_slice,Vmx):
    
    """
    Implementation of the simplex optimisation, for on slice,  using scipy
    
    x0 : initial parameters of the slice
    
    hyperparameters : parameters used for optimisation
    
    listSlice : set of slices
    
    grid_slices : matrix of cost
    
    set_o : set of outliers slices
    
    i_slice : slice of interest
    
    """
    
    
    
    #slicei=listSlice[i_slice]

    #slicei.set_parameters(x0)
    delta,xatol,fatol,lamb=hyperparameters[0],hyperparameters[1],hyperparameters[2],hyperparameters[4]
    
    #p=zeros(5) 
    p=(ones((x0.size,x0.size))*x0.T)+eye(6)*delta
    initial_s=concatenate(([x0],p),axis=0)


    NM = minimize(cost_fct,x0,args=(i_slice,listSlice,grid_slices,set_o,lamb,Vmx),method='Nelder-Mead',options={"disp" : False, "maxiter" : fatol, "maxfev":1e3, "xatol" : xatol, "initial_simplex" : initial_s , "adaptive" :  False})
   
    x_opt = NM.x #best parameter obtains after

    
    listSlice[i_slice].set_parameters(x_opt)
    

    ge,gn,gi,gu=[grid_slices[i,:,:] for i in range(0,4)]


    updateCostBetweenAllImageAndOne(i_slice,listSlice,ge,gn,gi,gu)
    grid_slices = array([ge,gn,gi,gu])
    print('dice',cost_from_matrix(gi,gu,set_o,i_slice),'mse',cost_from_matrix(ge,gn,set_o,i_slice))

        
    return x_opt




def mask_proportion(mask):
    
    x,y,z = mask.shape
    in_mask=0
    if x !=0 and y !=0 :
        for i in range(0,x):
            for j in range(0,y):
                if mask[i,j]>0:
                    in_mask=in_mask+1         
        res = in_mask/(x*y)
    else :
        res=0
    return res

   
 
    
def get_index(orientation,index_slice,listSlice):
    index_list = -1
    for i_slice in range(0,len(listSlice)):
        slicei = listSlice[i_slice]
        if slicei.get_orientation() == orientation and slicei.get_index_slice()==index_slice:
            index_list = listSlice.index(slicei)
            break
    return index_list


def cost_slice(grid_numerator,grid_denumerator,i_slice): #dice or mse

    num = somme(grid_numerator[:,i_slice])+somme(grid_numerator[i_slice,:])
    denum = somme(grid_denumerator[:,i_slice])+somme(grid_denumerator[i_slice,:])
    if denum==0:
        return 0
    
    return num/denum

def distance_to_center(stack,i_slice_in_stack,icenter): #distance between the slice and the center of the slice in the stack
    
    res=i_slice_in_stack-icenter
    if  res> 0:
        res = abs(res)/(len(stack)-icenter)
    else :
        res = abs(res)/icenter
    return res

def mask_difference(grid_union,grid_inter,i_slice):
    
    union = somme(grid_union[:,i_slice])+somme(grid_union[i_slice,:])
    #print(union)
    inter = somme(grid_inter[:,i_slice])+somme(grid_inter[i_slice,:])
    #print(inter)
    n=somme(grid_union[:,i_slice]>0)+somme(grid_union[i_slice,:]>0)
    #print(n)
    if n==0:
        return 0
    return (union-inter)/n

def std_intensity(data,mask):
    
    image=data*mask
    std_res=std(image)
    
    return std_res

def slice_center(mask3D):
    
    index = where(mask3D>0)
    center = sum(index,axis=1)/(somme(mask3D))
    centerw = concatenate((center[0:3],array([1])))
    icenter=ceil(centerw[2])
       
    return icenter

def std_volume(image,masks):
    
    data = [reshape(slicei.get_slice().get_fdata(),-1) for slicei in image]
    data = concatenate(data)
    mask_data = [reshape(mask,-1) for mask in masks]
    mask_data = concatenate(mask_data)
    brain_data = data*mask_data
    res = std(brain_data)
    return res

#update feature for each slices :
def update_feature(listSlice,listSliceError,grid_error,grid_nbpoint,grid_inter,grid_union):
    
    images,masks=createVolumesFromAlist(listSlice)
    variance = [compute_noise_variance(img) for img in images]
    index = [len(masks[n]) for n in range(0,len(masks))]
    pmasktot=[max(sum(masks[n][0:index[n]],axis=(1,2))) for n in range(0,len(masks))]
    mask_volume=[concatenate(masks[n][0:index[n]],axis=2) for n in range(0,len(masks))]
    center_volume = [slice_center(mask_volume[n]) for n in range(0,len(images))]
    std_total = [std_volume(images[n],masks[n]) for n in range(0,len(masks))]
    
    index_list = range(0,len(listSlice))
    index_list = array(index_list)
    for i_slice in range(0,len(listSlice)):
        currentError = listSliceError[i_slice]
        slicei=listSlice[i_slice]
        orientation=slicei.get_index_image()
        var=variance[orientation]
        #compute features : 
        
        #mse :
        stack = images[orientation]
        i_in_stack = stack.index(slicei)
        mse=0
        for i_stack in range(0,len(images)):
            if i_stack != orientation:
                var2 = variance[i_stack]
                interested_values  = [(slicei.get_index_image()==orientation or slicei.get_index_image()==i_stack) for slicei in listSlice]
                #tmp_list = concatenate((stack,images[i_stack]))
                #ge,gn,gi,gu = registration.computeCostBetweenAll2Dimages(tmp_list)
                interested_values = array(interested_values)
                zeros_values = index_list[where(interested_values==False)[0]]
                mse_tmp=compute_mse(i_slice,listSlice,zeros_values)
                #print('Variance ', var, var2)
                mse = mse + (1/(var+var2))*mse_tmp #1/(var+var2)*
        #print('mse',mse)
        #variance=compute_noise_variance(slicei)
        #mse=cost_slice(grid_error,grid_nbpoint,i_slice)
        currentError.set_mse(mse)
        
        
        #dice :
        dice=cost_slice(grid_inter,grid_union,i_slice)
        currentError.set_dice(dice)
        
        #difference in mm
        inter=mask_difference(grid_union,grid_inter,i_slice)
        currentError.set_inter(inter)
        
        #mask proportion
        orientation=slicei.get_index_image()
        mtot=pmasktot[orientation]
        mask=slicei.get_mask()
        mprop=somme(mask)
        currentError.set_mask_point(mprop)
        currentError.set_mask_proportion(mprop/mtot)
        
        #mask distance
        stack=images[orientation]
        i_slice_in_stack = [stack[index_slice].get_index_slice()==slicei.get_index_slice() for index_slice in range(0,len(stack))].index(True)
        center_dist = distance_to_center(stack,i_slice_in_stack,center_volume[orientation])
        currentError.set_center_distance(center_dist)
        
        #std_intensity:
        data=slicei.get_slice().get_fdata()
        std_in_image=std_intensity(data,mask)
        std_norm = std_total[orientation]
        currentError.set_std_intensity(std_in_image/std_norm)
        
        #ncc
        #ncc_var = compute_ncc(i_slice,listSlice)
        #currentError.set_ncc(ncc_var)

        #nb_point
        nbpoint = somme(grid_nbpoint[:,i_slice])+somme(grid_nbpoint[i_slice,:])
        currentError.set_nbpoint(nbpoint)
        
#Reference: J. Immerkær, “Fast Noise Variance Estimation”, Computer Vision and Image Understanding, Vol. 64, No. 2, pp. 300-302, Sep. 1996
def compute_noise_variance(volume):
    
    data=[slicei.get_slice().get_fdata().squeeze() for slicei in volume]
    #data = volume.get_slice().get_fdata().squeeze()
    #H,W = data.shape
    mask=[slicei.get_mask().squeeze() for slicei in volume]
    #mask = volume.get_mask().squeeze()
    laplacien=array([[0,1,0],[1,-4,1],[0,1,0]]) #laplacien filter
    #
    laplacien_convolution = [scipy.ndimage.convolve(data_slice*mask_slice,laplacien).reshape(-1) for (data_slice,mask_slice) in zip(data,mask)] #convolve with laplacien
    #laplacien_convolution = scipy.ndimage.convolve(data*mask,laplacien).reshape(-1)
    
    data_mask = [mask_slice.reshape(-1) for mask_slice in mask]
    data_mask = concatenate(data_mask)
     
    vect = concatenate(laplacien_convolution)
    #vect = laplacien_convolution
    vect=vect[data_mask>0]
    #print(size(vect))
    med = np.median(vect)
    #print(vect)
    #print(var(vect)/20)
    
    mad = np.median(abs(vect - med))
    k=1.4826
    sigma=(k*mad)
    variance=sigma**2
        
    return variance/20


def compute_ncc(i_slice,listSlice):
    
    ncc_moy=[]
    slicei=listSlice[i_slice]
    for slice2 in listSlice:
        if slice2 != slicei:
            _,_,_,_,ncc,_ = costLocal(slicei,slice2)
            ncc_moy.append(ncc)
    ncc_moy=array(ncc_moy)
    ncc_moy>0
    n = somme(ncc_moy>0)
    #print(n)
    ncc_moy=somme(ncc_moy[ncc_moy>0])
    
    return ncc_moy/n

def compute_mse(i_slice,listSlice,zeros_index):
    
    mse_moy=[]
    nbpoint=[]
    slicei=listSlice[i_slice]
    for i_slice2 in range(0,len(listSlice)):
        if not i_slice2 in zeros_index:
            slice2=listSlice[i_slice2]
            if slice2 != slicei:
                _,_,_,union,_,mse = costLocal(slicei,slice2)
                #print('test',mse[0]/mse[1])
                mse_moy.append(mse[0])
                nbpoint.append(mse[1])
    mse_moy=array(mse_moy)
    nbpoint=array(nbpoint)
    
    mse_moy=somme(mse_moy[mse_moy>-1])
    #print('mse_moy',mse_moy/sum(nbpoint[nbpoint>0]))
    if somme(nbpoint[nbpoint>-1])==0:
        return nan
    
    return mse_moy/somme(nbpoint[nbpoint>-1])
        
def data_to_classifier(listSliceError,features,bad_slices_threshold):
    
    error=array([currentError.get_error() for currentError in listSliceError])
    Y=error>bad_slices_threshold
    
    nb_features=len(features)
    nb_point=len(listSliceError)
    X=zeros((nb_point,nb_features))
    
    for i_feature in range(0,len(features)):
        fe = features[i_feature]
        vaules_features = array([getattr(currentError,'_'+fe) for currentError in listSliceError])
        vaules_features=np.squeeze(vaules_features)
        print(vaules_features.shape)
        vaules_features[isnan(vaules_features)]=0
        vaules_features[isinf(vaules_features)]=0
        X[:,i_feature]=vaules_features
    
    return X,Y #X contains the features for all points and Y a classification of bad and good slices
        
def detect_misregistered_slice(listSlice,grid_slices,loaded_model):
     
     ge = grid_slices[0,:,:]
     gn = grid_slices[1,:,:]
     gi = grid_slices[2,:,:]
     gu= grid_slices[3,:,:]
     
     
     
     listErrorSlice = [ErrorSlice(slicei.get_index_image(),slicei.get_index_slice()) for slicei in listSlice]
     update_feature(listSlice,listErrorSlice,ge,gn,gi,gu)
     features=['mse','inter','dice','mask_proportion','std_intensity']
     X,Y=data_to_classifier(listErrorSlice,features,1.5)
     out_image,listError = detect_slices_out_of_images(listErrorSlice,gi)
     
     estimated_y=loaded_model.predict(X)
     #estimated_y = array([X[:,2]<0.9])
     #estimated_y = estimated_y[0]
     
     return abs(estimated_y),abs(out_image)    

def detect_slices_out_of_images(listSliceError,intersection):
    #list_far_far_away=[]
    nbSlice=len(listSliceError)
    list_far_far_away=zeros(nbSlice)
    #print('nbSlice',nbSlice)
    i_slice=0
    index=0
    while i_slice<(nbSlice):
        #nbpoint = listSliceError[index].get_nbpoint()
        nbpoint = somme(intersection[:,i_slice])+somme(intersection[i_slice,:])
        #print(nbpoint)
        mask = listSliceError[index].get_mask_point()
        #print(mask)
        if  nbpoint < mask/2 :
            far_far_away = (listSliceError[index].get_orientation(),listSliceError[index].get_index())
            #list_far_far_away.append(far_far_away)
            list_far_far_away[i_slice]=1
            listSliceError.remove(listSliceError[index])
            index=index-1
            
        index=index+1
        i_slice=i_slice+1
    return list_far_far_away,listSliceError

@jit(nopython=True,fastmath=True)
def somme(index):
    return sum(index)

@jit(nopython=True,fastmath=True)
def line(start,stop,num):
    return linspace(start,stop,num)
