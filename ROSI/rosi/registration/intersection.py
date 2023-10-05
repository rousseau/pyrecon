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
from .tools import line, separate_slices_in_stacks,apply_gaussian_filtering, computeMaxVolume, somme
from .transformation import ParametersFromRigidMatrix, rigidMatrix
from numba import jit,njit
from multiprocessing import Pool
from functools import partial
import pickle
from .outliers_detection.outliers import sliceFeature, separate_features_in_stacks
from scipy.optimize import Bounds
from .sliceObject import SliceObject



@jit(nopython=True,fastmath=True)
def line_intersection_in_world(M_k : array((4,4)),M_kprime : array((4,4))) -> (array((3,2)),array((3,2)),int) : 
    """
    Compute the intersection line between two planes. The result is given in world coordinate sytem
    """  
    
    #normal vector to the 0xy plan
    n_k = cross(M_k[0:3,0],M_k[0:3,1]) 
    if norm(n_k)<1e-6: #no division by 0, case M11 // M12 (normally that souldn't be the case)
        return array([float_(0.0)]),array([float_(0.0)]),array([float_(0.0)])
    nnorm_k = n_k/norm(n_k)
    n_k = nnorm_k
    t_k = M_k[0:3,3]

    
    n_kprime = cross(M_kprime[0:3,0],M_kprime[0:3,1]) 
    if norm(n_kprime)<1e-6: #no division by 0, case M21 // M22 (normally that souldn't be the case)
        return array([float_(0.0)]),array([float_(0.0)]),array([float_(0.0)])
    nnorm_kprime = n_kprime/norm(n_kprime)
    n_kprime = nnorm_kprime
    t_kprime = M_kprime[0:3,3]

    
    alpha = ascontiguousarray(n_k) @ ascontiguousarray(n_kprime) #if the vector are colinear alpha will be equal to one (since n1 and n2 are normalized), can happend if we consider two parralel slice
    beta =  ascontiguousarray(n_k) @ ascontiguousarray(t_k)
    gamma = ascontiguousarray(n_kprime) @ ascontiguousarray(t_kprime)   
    
    if abs((1 - alpha*alpha))<1e-6: #if the vector are colinear, there is no intersection
        return array([float_(0.0)]),array([float_(0.0)]),array([float_(0.0)])
    a = 1/(1 - alpha*alpha)
    g = a*(beta - alpha*gamma)
    h = a*(gamma - alpha*beta)

    #line equation
    coeff = cross(n_k,n_kprime)
    pt = g*n_k + h*n_kprime

    return coeff, pt, array([float_(1.0)])
 
@jit(nopython=True,fastmath=True)  
def segment_intersection_in_world(slice_k : array,
                                  M_k : array((4,4)),
                                  coeff : array((3,2)),
                                  pt : array((3,2))) -> (array((2,1)),int):
    """
    Compute the segment of intersection between the line and the 2D slice. The result is given in world coordinate system.
    """
    
    #line equation into the image plan    
    Minv = inv(M_k)
    rinv = inv(M_k[0:3,0:3])
    n = ascontiguousarray(rinv) @ ascontiguousarray(coeff)
    #print(coeff,n)
    pt = concatenate((pt,array([1]))) 
    ptimg = ascontiguousarray(Minv) @ ascontiguousarray(pt)
    #print(pt,ptimg)
    
    a = -n[1]
    b = n[0]
    c = -(a*ptimg[0] + b*ptimg[1])
    
    
    #Intersection with the plan
    coordinate_in_image = zeros((4,2)) #2 points of intersection of coordinates i,j,k 
    coordinate_in_image[3,:] = ones((1,2))
    width = slice_k.shape[0]-1
    height = slice_k.shape[1]-1
    #print(width)
    #print(height)
    #print(a,b,c)

    indice=0
    #The intersection on a corner are considered only once
    
    if (abs(a)>1e-10): #if a==0, the division by zeros in not possible, in this case we have only two intersection possible : 
           
        i=(-c/a); j=0
        
        if  i >= 0 and i < width: #if y=0 x=-c/a  #the point (0,0) is considered here
            coordinate_in_image[0,indice] =  i
            coordinate_in_image[1,indice] =  j
            indice=indice+1
        
        i=((-c-b*height)/a); j=height
       
        if (i>0) and (i <= width) : #if y=height x=-(c-b* height)/a #the point  (width,height) is considered here
            coordinate_in_image[0,indice] = i
            coordinate_in_image[1,indice] = j
            indice=indice+1
        
         
    if (abs(b)>1e-10): #if b==0, the divistion by zeros in not possible, in this case we have only two intersection possible :
           
        i=0; j=(-c/b);
        
        if j>0 and  j <= height: #if x=0 y=-c/b #the point (0,heigth) is considered here
            coordinate_in_image[0,indice] = i 
            coordinate_in_image[1,indice] = j
            indice=indice+1
       
        i=width; j=(-c-a*width)/b
        
        if j>=0  and j<height: #if x=width y=(-c-a*width)/b  #the point (width,0) is considered here
            coordinate_in_image[0,indice] = i
            coordinate_in_image[1,indice] = j
            indice=indice+1

    
    if indice < 2 or indice > 2:
        return array([float_(0.0)]),array([float_(0.0)])
        
    #Compute the intersection point coordinates in the 3D space
    coordinate_in_world = zeros((4,2)) #2 points of intersection, with 3 coordinates x,y,z
    coordinate_in_world[3,:] = ones((1,2))
    coordinate_in_world = ascontiguousarray(M_k) @ ascontiguousarray(coordinate_in_image) 
    
    coordinate_in_world[0:3,0] = coordinate_in_world[0:3,0] - pt[0:3]
    coordinate_in_world[0:3,1] = coordinate_in_world[0:3,1] - pt[0:3]
    
    squareNorm = ascontiguousarray(coeff) @ ascontiguousarray(coeff.transpose())
    segment_world = ascontiguousarray((((1/squareNorm) * coeff.transpose())) @ ascontiguousarray(coordinate_in_world[0:3,:])) 

    
    return segment_world,array([float_(1.0)])


    
@jit(nopython=True,fastmath=True)    
def segment_union(segment_world_k : array((2,1)),
                  segment_world_kprime : array((2,1)),
                  type_estimation : str = 'intersection') -> array((2,1)): 
    
    """
    Compute the common segment between two images
    Correspond to equations : 
    if type_estimation = 'intersection' : 
        M_k(v) != NaN and M_k'(v') != NaN
    if type_estimation = 'union' :
        M_k(v) != NaN or M_k'(v') != NaN
    """
    if type_estimation=='intersection':
        
        seg_union = zeros(2)
        
        seg_union[0] = max(np.min(segment_world_k),np.min(segment_world_kprime))
        seg_union[1] = min(np.max(segment_world_k),np.max(segment_world_kprime))
    
    else :
        
            if type_estimation =='union':
                seg_union = zeros(2)
                seg_union[0] = min(np.min(segment_world_k),np.min(segment_world_kprime))
                seg_union[1] = max(np.max(segment_world_k),np.max(segment_world_kprime))
            else :
                print('inter must be intersection or union')
                
    return seg_union 

#@jit(nopython=True)
def common_segment_in_image(slice_k : SliceObject,
                            M_k : array((4,4)),
                            slice_kprime : array,
                            M_kprime : SliceObject,
                            resolution : array(float)) :
    """
    Compute the coordinates of the two extremity points of the segment in the 2 image plans
    The function compute : 
        set of point v and v' such as Mest_k(v)=Mest_k'(v') 
        and return : 
        v,v',s_k(v),s_k'(v'),m_k(v),m_k'(v'),ok,ok
    """
    

    data_k = slice_k.get_slice().get_fdata()
    data_kprime = slice_kprime.get_slice().get_fdata()

    coeff,pt,ok=line_intersection_in_world(M_k,M_kprime)
    ok=int(ok[0])
    #print(ok)
    
    if ok<1: #if there is no intersection lines (the 2 planes are parralel) it is useless to compute the intersectionSegment
        return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)
        
    #sliceimage1=Slice1.get_slice().get_fdata()
    #print(coeff)
    word_segment_k,ok=segment_intersection_in_world(data_k,M_k,coeff,pt) #if there is no intersection segment (the line of intersection is outisde of the image or on a corner), it useless to compute a common segment
    ok1=int(ok[0])
    #print('seg :',ok1)
   
    #if ok<1:
    #    return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)
    
    #sliceimage2=Slice2.get_slice().get_fdata()
    word_segment_kprime,ok=segment_intersection_in_world(data_kprime,M_kprime,coeff,pt)
    ok2=int(ok[0])
    
    if ok1<1 and ok2<1:
        return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)
    
    if ok1<1 or ok2<1: #if we have a segment of intersection for both images
        
        if ok1<1:
            seg_union=word_segment_kprime
        else:
            seg_union=word_segment_k
        
        coordinates_union_in_world = zeros((3,2))
            
        coordinates_union_in_world[0:3,0] = seg_union[0] * coeff + pt #Point corresponding to the value of lambda
        coordinates_union_in_world[0:3,1] = seg_union[1] * coeff + pt
            
        coordinates_union_in_world = concatenate((coordinates_union_in_world,array([[1,1]])))
            
        set_v = zeros((4,2))
        set_v[3,:] = ones((1,2))
                        
        set_vprime = zeros((4,2))
        set_vprime[3,:] = ones((1,2))
                        
        set_v = ascontiguousarray(inv(M_k)) @ ascontiguousarray(coordinates_union_in_world) #there is a point, it simply not belong to the image but it's in the plan

        set_vprime = ascontiguousarray(inv(M_kprime)) @ ascontiguousarray(coordinates_union_in_world)

        if not equal(set_v[0:2,0],set_v[0:2,1]).all() :
            segment_len_ink = norm(set_v[0:2,0] - set_v[0:2,1])
        else :
            segment_len_ink=0
 
        if not equal(set_vprime[0:2,0],set_vprime[0:2,1]).all():
            segment_len_inkprime = norm(set_vprime[0:2,0]- set_vprime[0:2,1])
        else:
            segment_len_inkprime=0
        if segment_len_ink + segment_len_inkprime == 0:
            return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)

        nbpoint=int((round(segment_len_ink+segment_len_inkprime)+1)/resolution)
    
        profile_k,m_kv,nb_point_in_k=interplolation_inimage(slice_k,set_v,nbpoint)
        profile_kprime,m_kvprime,nb_point_in_kprime=interplolation_inimage(slice_kprime,set_vprime,nbpoint)
 
        profile_k = concatenate((profile_k[m_kv],zeros(nb_point_in_kprime)))
        profile_kprime = concatenate((zeros(nb_point_in_k),profile_kprime[m_kvprime]))
        m_kv = concatenate((m_kv[m_kv],zeros(nb_point_in_kprime)))
        m_kvprime = concatenate((zeros(nb_point_in_k),m_kvprime[m_kvprime]))
        
        nbpoint=nb_point_in_k+nb_point_in_kprime
        if nbpoint==0:
            return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)
        
    elif max(word_segment_k)>min(word_segment_kprime) and max(word_segment_kprime)>min(word_segment_k):
            
            seg_union_in_world = segment_union(word_segment_k,word_segment_kprime,'union')
        
            if seg_union_in_world[0]==seg_union_in_world[1]: #the segment is nul, there is no intersection
                return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)
                
            coordinates_union_in_world = zeros((3,2))
            
            coordinates_union_in_world[0:3,0] = seg_union_in_world[0] * coeff + pt #Point corresponding to the value of lambda
            coordinates_union_in_world[0:3,1] = seg_union_in_world[1] * coeff + pt
            
            coordinates_union_in_world = concatenate((coordinates_union_in_world,array([[1,1]])))
            
            set_v = zeros((4,2))
            set_v[3,:] = ones((1,2))
            
            
            set_vprime = zeros((4,2))
            set_vprime[3,:] = ones((1,2))
           
            
            set_v = ascontiguousarray(inv(M_k)) @ ascontiguousarray(coordinates_union_in_world)

            set_vprime = ascontiguousarray(inv(M_kprime)) @ ascontiguousarray(coordinates_union_in_world) 
            
            segment_len_ink = norm(set_v[0:2,0] - set_v[0:2,1]) #distance between two points on the two images
            segment_len_inkprime = norm(set_vprime[0:2,0] - set_vprime[0:2,1]) 

            #res = min(Slice1.get_slice().header.get_zooms())
            #the smaller resolution of a voxel
                
            if resolution<0: #probmem with the resolution of the image
                return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)        
            if max(segment_len_ink,segment_len_inkprime)<1: #no pixel in commun
                return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)
            
            nbpoint = int(round(max(segment_len_ink,segment_len_inkprime)+1)/resolution) #choose the max distance and divide it by the smaller resolution 

            
            if ok>0:
                profile_k,m_kv,nb_point_in_k=interplolation_inimage(slice_k, set_v, nbpoint)  #profile in slice 1
                #print(index1)
                profile_kprime,m_kvprime,nb_point_in_kprime=interplolation_inimage(slice_kprime, set_vprime, nbpoint)  #profile in slice 2
                #print(index2)
    else:

        coordinates_union_in_world = zeros((3,2))
            
        coordinates_union_in_world[0:3,0] = word_segment_k[0] * coeff + pt #Point corresponding to the value of lambda
        coordinates_union_in_world[0:3,1] = word_segment_k[1] * coeff + pt
            
        coordinates_union_in_world = concatenate((coordinates_union_in_world,array([[1,1]])))
            
        set_v = zeros((4,2))
        set_v[3,:] = ones((1,2))
                        
        set_vprime = zeros((4,2))
        set_vprime[3,:] = ones((1,2))
                        
        set_v = ascontiguousarray(inv(M_k)) @ ascontiguousarray(coordinates_union_in_world)

        set_vprime = ascontiguousarray(inv(M_kprime)) @ ascontiguousarray(coordinates_union_in_world)


        if not equal(set_v[0:2,0],set_v[0:2,1]).all() :
            segment_len_ink = norm(set_v[0:2,0] - set_v[0:2,1])
        else :
            segment_len_ink=0
 
        if not equal(set_vprime[0:2,0],set_vprime[0:2,1]).all():
            segment_len_inkprime = norm(set_vprime[0:2,0]- set_vprime[0:2,1])
        else:
            segment_len_inkprime=0
        if segment_len_ink + segment_len_inkprime == 0:
            return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)

        nbpoint=int((round(segment_len_ink+segment_len_inkprime)+1)/resolution)
    
        profile_k,m_kv,nb_point_in_k=interplolation_inimage(slice_k,set_v,nbpoint)
        profile_kprime,m_kvprime,nb_point_in_kprime=interplolation_inimage(slice_kprime,set_vprime,nbpoint)
 
        profile_k = concatenate((profile_k[m_kv],zeros(nb_point_in_kprime)))
        profile_kprime = concatenate((zeros(nb_point_in_k),profile_kprime[m_kvprime]))
        m_kv = concatenate((m_kv[m_kv],zeros(nb_point_in_kprime)))
        m_kvprime = concatenate((zeros(nb_point_in_k),m_kvprime[m_kvprime]))
        
        nbpoint=nb_point_in_k+nb_point_in_kprime
        if nbpoint==0:
            return zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_),zeros((2,2),dtype=float_)


    return set_v,set_vprime,profile_k,profile_kprime,m_kv,m_kvprime,(nbpoint)*ones((2,2),dtype=float_),ones((2,2),dtype=float_)


def interplolation_inimage(slice_k : array,
                           segment_inimage : array((3,2)),
                           nbpoint : int):
    """
    Interpol values on the segment to obtain the profil intensity
    The function return s_k(v) and m_k(v) with v the set of points such as  : Mest_k(v)=Mest_k(v') 
    """
    
    if nbpoint == 0:
        return 0,0,0
    
    profile= zeros(nbpoint)
    interpolMask = zeros(nbpoint)
    v_points = zeros((3,nbpoint))
    v_points[0,:] = line(segment_inimage[0,0],segment_inimage[0,1],nbpoint)
    v_points[1,:] = line(segment_inimage[1,0],segment_inimage[1,1],nbpoint)
   
    
    mask = slice_k.get_mask()
    map_coordinates(slice_k.get_slice().get_fdata(), v_points , output=profile, order=1, mode='constant', cval=0, prefilter=False)
    map_coordinates(mask, v_points, output=interpolMask, order=0, mode='constant',cval=0,prefilter=False)
    
    m_kv =~isnan(profile) * interpolMask>0

      
    return profile,m_kv,somme(m_kv)


def common_profile(profile_k : array,
                   m_kv : array,
                   profile_kprime : array,
                   m_kvprime : array,
                   nbpoint : int,
                   type_estimation='union'):
    """
    Compute the intensity of points along the intersection in two orthogonal slice.
    The function return to s_k(v) and s_k'(v') with v and v' the set of points such as :
    Mest_k(v)=Mest_k(v') and m_k(v)=1 or m_k'(v')=1 (if type_estimation = 'union')
    Mest_k(v)=Mest_k(v') and m_k(v)=1 and m_k'(v')=1 (if type_estimation = 'intersection')
    """
    if nbpoint==0:
        return 0,0,0
    
    valindex=linspace(0,nbpoint-1,nbpoint,dtype=int)
    if type_estimation=='union':
        common_mask = m_kv+m_kvprime #m_k(v)=1 or m_k'(v')=1
    
    elif type_estimation=='intersection':
        common_mask = m_kv*m_kvprime #m_k(v)=1 ans m_k'(v')=1
    
    else:
        print(type_estimation, "is not recognized, choose either 'itersection' or 'union'")
    
    common_mask = valindex[common_mask==True]


    profile_k = zeros(profile_k.shape[0])  
    profile_k[~isnan(profile_k)] = profile_k[~isnan(profile_k)]
    profile_kprime = zeros(profile_kprime.shape[0])  
    profile_kprime[~isnan(profile_kprime)] = profile_kprime[~isnan(profile_kprime)] 
 
    return profile_k[common_mask],profile_kprime[common_mask],common_mask
    

@jit(nopython=True,fastmath=True)
def error(profile_k,profile_kprime):
    """
    Compute sumed squares error between two intensity profils. The function is accelerated with numba.
    The function compute : 
    S^2(k,k') = sum_v (s_k(v) - s_k'(v'))^2        
    """
    return somme((profile_k - profile_kprime)**2)
    
def NCC(profile_k,profile_kprime):
    
    x=profile_k[~isnan(profile_k)]
    y=profile_kprime[~isnan(profile_kprime)]
    mux = mean(x)
    muy = mean(y)
    stdx = std(x)
    stdy = std(y)

    ncc=-1
    if stdx>0 and stdy>0:
        ncc = (1/(len(x)-1))*somme((x-mux)*(y-muy))/(stdx*stdy)
        
    return ncc


def cost_between_2slices(slice_k,slice_kprime):
    """
    Compute the MSE between two slices
    """

    resolution=min(min(slice_k.get_slice().header.get_zooms(),slice_kprime.get_slice().header.get_zooms()))
    
    M_k=slice_k.get_estimatedTransfo();M_kprime=slice_kprime.get_estimatedTransfo()
    
    _,_,profile_k,profile_kprime,m_kv,m_kvprime,nbpoint,ok = common_segment_in_image(slice_k,M_k,slice_kprime,M_kprime,resolution)
    
    ok=int(ok[0,0]); nbpoint=int(nbpoint[0,0]) #ok and nbpoints are 2-size vectors to allow using numba with this function
    #print('seg :', ok)

    square_error=0; common_point=0; intersection=0; union=0; ncc_var=-1; mse_coupe=(-1,-1)
    #print(ok)
    
    if ok>0:
        
        #mask union : to comupte MSE
        union_mask=logical_or(m_kv,m_kvprime) #m_k(v)=1 or m_k'(v')=1
        union_profile_v=profile_k[union_mask] #s_k(v) for v such as Mest_k(v)=Mest_k'(v') and m_k(v)=1 or m_k'(v')=1
        union_profile_vprime=profile_kprime[union_mask] #s_k'(v') for v' such as Mest_k(v)=Mest_k'(v') and m_k(v)=1 or m_k'(v')=1
        
        #mask intersection : to compute DICE
        intersection_mask=logical_and(m_kv,m_kvprime) #m_k(v)=1 and m_k'(v')=1
        intersection_profile_v=profile_k[intersection_mask] #s_k(v) for v such as Mest_k(v)=Mest_k'(v') and m_k(v)=1 and m_k'(v')=1
        intersection_profile_vprime=profile_kprime[intersection_mask] #s_k'(v') for v' such as Mest_k(v)=Mest_k'(v') and m_k(v)=1 and m_k'(v')=1
        
        square_error=error(union_profile_v,union_profile_vprime)

        if len(intersection_profile_v) !=0:
            mse_coupe=(error(intersection_profile_v,intersection_profile_vprime),len(intersection_profile_v))
        
        common_point=len(union_profile_v) 
        intersection=len(intersection_profile_v) 
        point_slice_k=sum(m_kv)
        point_slice_kprime=sum(m_kvprime)
        union=point_slice_k+point_slice_kprime 
  
        
    return square_error,common_point,2*intersection,union,ncc_var,mse_coupe

def update_cost_matrix(index_slice : int,
                       listOfSlices : 'list[SliceObject]',
                       square_error_matrix : array,
                       nbpoint_matrix :array,
                       intersection_matrix : array,
                       union_matrix : array):
    """
    The function update the two matrix used to compute MSE and the two matrix used to compute dice, when one slice position is modified
    S^2[k,k'] = sum_v((s_k(v)-s_k"(v')) ^2) 1(m_k(v)=1 or m_v'(v')=1)
    N[k,k'] = sum_v 1(m_k(v)=1 or m_v'(v')=1)
    I[k,k'] = 2 * sum_v 1(m_k(v)=1 and m_v'(v')=1)
    U[k,k'] = sum_v 1(m_k(v) = 1) + sum_v' 1(m_k'(v') = 1)
    """

    slice_k=listOfSlices[index_slice] 
    
    kprime=0
    while kprime < index_slice :
      
        slice_kprime=listOfSlices[kprime]      
        if slice_k.get_stackIndex() != slice_kprime.get_stackIndex():
                square_error_matrix[index_slice,kprime],nbpoint_matrix[index_slice,kprime],intersection_matrix[index_slice,kprime],union_matrix[index_slice,kprime],_,_=cost_between_2slices(slice_k,slice_kprime)               #else:
        kprime+=1
    
    while kprime < len(listOfSlices):
        slice_kprime=listOfSlices[kprime]
        if slice_k.get_stackIndex() != slice_kprime.get_stackIndex():            
                square_error_matrix[kprime,index_slice],nbpoint_matrix[kprime,index_slice],intersection_matrix[kprime,index_slice],union_matrix[kprime,index_slice],_,_=cost_between_2slices(slice_kprime,slice_k)
        kprime+=1
    


def compute_cost_matrix(listOfSlice):
    """
    Computes matrix used in calcul of MSE and DICE.
    for each couple of slice, k,k' such as k>k': 
        S^2[k,k'] = sum_v((s_k(v)-s_k"(v')) ^2) 1(m_k(v)=1 or m_v'(v')=1)
        N[k,k'] = sum_v 1(m_k(v)=1 or m_v'(v')=1)
        I[k,k'] = 2 * sum_v 1(m_k(v)=1 and m_v'(v')=1)
        U[k,k'] = sum_v 1(m_k(v) = 1) + sum_v' 1(m_k'(v') = 1)
    """
   
    #Initialization
    number_slices = len(listOfSlice)
    square_error_matrix=zeros((number_slices,number_slices))
    nbpoint_matrix=zeros((number_slices,number_slices))
    intersection_matrix=zeros((number_slices,number_slices))
    union_matrix=zeros((number_slices,number_slices))
    k=0
    

    for k in range(number_slices): 
        slice_k=listOfSlice[k]
        for kprime in range(number_slices):
            slice_kprime=listOfSlice[kprime]
            if (k > kprime):
                if slice_k.get_stackIndex() != slice_kprime.get_stackIndex():
                    square_error,nbpoint,intersection,union,_,_=cost_between_2slices(slice_k,slice_kprime) #computed cost informations between two slices
                    square_error_matrix[k,kprime]=square_error 
                    nbpoint_matrix[k,kprime]=nbpoint
                    intersection_matrix[k,kprime]=intersection
                    union_matrix[k,kprime]=union
                    
    return square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix     


@jit(nopython=True)
def compute_cost_from_matrix(numerator,denumerator):
    
    """
    Compute the cost on all slices from two matrix. Can be equally used to compute the MSE and the DICE
    """

    sum_num = somme(numerator)
    sum_denum= somme(denumerator)
    
    if sum_denum>0:
        cost = sum_num/sum_denum
    
    else:
        cost=0
    
    return cost

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
    grid_outliers=zeros((nbslice,nbslice))
   
   
    numerator = sum(grid_numerator_no_o)# * grid_outliers)
    denumerator = sum(grid_denumerator_no_o)# * grid_outliers)

    if denumerator==0:
        cost=nan
    cost=numerator/denumerator
    
    return cost

def cost_fct(x0,k,listOfSlice,cost_matrix,set_o,lamb,Vmx):
    """
    function we want to minimize.
    """
    
    square_error_matrix = cost_matrix[0,:,:]
    nbpoint_matrix = cost_matrix[1,:,:]
    intersection_matrix = cost_matrix[2,:,:]
    union_matrix = cost_matrix[3,:,:]
    
    x = copy(x0) #copy to use bound in miminization fonction
    #print(x)
    slicei = listOfSlice[k]
    slicei.set_parameters(x)
    #print(x)
    
    update_cost_matrix(k,listOfSlice,square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix)
    

    cost_matrix = array([square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix])
   
    #print(type(set_o[0]))
    #set_o = zeros((len(listSlice)))
    mse = cost_from_matrix(square_error_matrix,nbpoint_matrix,set_o,k)

    nbslice = shape(intersection_matrix)[0]
    i_slice1 = linspace(0,nbslice,nbslice,dtype=int)
    i_slice2 = linspace(0,nbslice,nbslice,dtype=int)
    index=np.meshgrid(i_slice1,i_slice2)
    bool_ind=index[0]<index[1]
    dice = sum(intersection_matrix[bool_ind])
    dice=dice/Vmx
    #print(dice)
    #dice = cost_from_matrix(gi,gu,set_o,i_slice)
    #print('nbslice',i_slice,'mse',mse,'dice',dice)
    #print(dice/Vmx)

    cost = mse - lamb*(dice)
    
    return cost


    

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
        dicRes["evolutiontransfo"].extend(slicei.get_estimatedTransfo()) #evolution of the global matrix applied to the image







   








