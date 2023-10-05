#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 16:38:29 2022

@author: mercier

This script aims to create motion simulation on an MRI image to validate the registration algorithm

"""
import sys
sys.path.append("..registration")

import numpy as np
import random as rd
from ..registration.intersection import common_segment_in_image 
#import common_segment_in_image
from scipy.ndimage import map_coordinates
from scipy.ndimage import distance_transform_cdt
from ..registration.tools import separate_slices_in_stacks 
from ..registration.outliers_detection.outliers import sliceFeature
from ..registration.outliers_detection.outliers import separate_features_in_stacks
from ..registration.sliceObject import SliceObject
import joblib
from nibabel import Nifti1Image


    
def tre_indexes(stack_fk : 'list[SliceObject]',
                stack_fkprime : 'list[SliceObject]',
                M_fk : np.array,
                M_fkprime : np.array,
                rejectedSlices : list) -> (np.array,np.array):
    
    """
    Take two volumes, with simulated motion, and compute the common points of the motion corrected volume : ie points  points v and v' from slices k and k' such as M_k(v) = M_k'(v')
    """
    
    set_v = []
    set_vprime = []
   
    for zk in range(len(stack_fk)):
        
        k = stack_fk[zk]
        mk = k.get_mask()
        
        for zkprime in range(len(stack_fkprime)):
            kprime = stack_fkprime[zkprime]
            m_kprime = kprime.get_mask()
            
          
            if ((k.get_stackIndex,k.get_indexSlice()) not in rejectedSlices) and ((kprime.get_stackIndex,kprime.get_indexSlice()) not in rejectedSlices) :
              
                Mk=M_fk[stack_fk[zk].get_indexSlice(),:,:] @ k.get_slice().affine  #equivalent to M_k
                Mk_prime=M_fkprime[stack_fkprime[zkprime].get_indexSlice(),:,:] @ kprime.get_slice().affine  #equivalent to M_k'
                resolution=min(min(k.get_slice().header.get_zooms(),kprime.get_slice().header.get_zooms()))
                v_temp,vprime_temp,_,_,_,_,nbpoint,ok = common_segment_in_image(k,Mk,kprime,Mk_prime,resolution) #list of point v and v' such as M_k(v) = M_k'(v') (1)

                if ok[0,0]>0: #make sure there exist some point v and v' that satisfied equation (1) 
                    #look for the point that also satisfied : mk(v)=mk'(v')=1

                    nbpoint=np.int32(nbpoint[0,0]);ok=np.int32(ok[0,0])
                    dist = 1
                    nbpt = int(np.ceil(nbpoint/dist)) ##???
        
                    v = np.zeros((nbpt,3))
                    v[:,0] = np.linspace(v_temp[0,0],v_temp[0,1],nbpt)
                    v[:,1] = np.linspace(v_temp[1,0],v_temp[1,1],nbpt)
                    v[:,2] = np.ones(nbpt)*zk
                    output_mk = np.zeros(nbpt)
                    v_in_w = np.zeros((3,nbpt))
                    v_in_w[0,:] = v[:,0]
                    v_in_w[1,:] = v[:,1]
                    v_in_w[2,:] = np.zeros(nbpt)
                    
                    vprime = np.zeros((nbpt,3))
                    vprime[:,0] = np.linspace(vprime_temp[0,0],vprime_temp[0,1],nbpt)
                    vprime[:,1] = np.linspace(vprime_temp[1,0],vprime_temp[1,1],nbpt)
                    vprime[:,2] = np.ones(nbpt)*zkprime
                    output_mkprime = np.zeros(nbpt)
                    vprime_in_w = np.zeros((3,nbpt))
                    vprime_in_w[0,:] = vprime[:,0]
                    vprime_in_w[1,:] = vprime[:,1]
                    vprime_in_w[2,:] = np.zeros(nbpt)
                    
                    map_coordinates(mk, v_in_w, output=output_mk, order=0, mode='constant',cval=np.nan,prefilter=False)
                    map_coordinates(m_kprime, vprime_in_w, output=output_mkprime, order=0, mode='constant',cval=np.nan,prefilter=False)
                    
                    common_mask = [output_mk[index] or output_mkprime[index] for index in range(0,nbpt)]

                    
                    #make sure that you are data without motion : check if the mask are identiqual
                    #if not (output_mk[0:nbpt]==output_mkprime[0:nbpt]).all():
                    #    raise Exception('Check inpus image or transformation, the points are not the same')
                    
                    common_mask=np.array(common_mask,dtype=bool)
                    
                    arrayv=v[common_mask] #v points such as M_k(v) = M_k'(v') and mk(v)=mk'(v')
                    listpoint = arrayv.tolist()
                    set_v.extend(listpoint)
                        
                    arrayvprime=vprime[common_mask] #v' points such as M_k(v) = M_k'(v') and mk(v)=mk'(v')
                    listpoint = arrayvprime.tolist()
                    set_vprime.extend(listpoint)
            
    return np.array(set_v),np.array(set_vprime)


    
def tre(set_v : np.array,
        set_vprime : np.array,
        stack_fk : 'list[SliceObject]',
        stack_fkprime : 'list[SliceObject]') -> np.array:

    """
    Take two volume with simulated motion and compute the TRE for each point v and v' (such as M_k(v)=M_k(v') and mk(v)=mk(v')) of the two volume
    """
    
    nbpoint = set_v.shape[0]
    set_tre = np.zeros(nbpoint)
    
    for index in range(nbpoint):
        
        v = set_v[index,:]
        
        k = int(v[2])
        print(k)

        Mest_k = stack_fk[k].get_transfo()
        homogeneous_v = np.zeros(4)
        homogeneous_v[0:2] = v[0:2]
        homogeneous_v[3] = 1
        v_in_world = Mest_k @ homogeneous_v #Mest_k(v)
        
        vprime = set_vprime[index,:]
        
        kprime = int(vprime[2])

        Mest_kprime = stack_fkprime[kprime].get_transfo()
        homogeneous_vprime = np.zeros(4)
        homogeneous_vprime[0:2] = vprime[0:2]
        homogeneous_vprime[3] = 1
        vprime_in_world = Mest_kprime @ homogeneous_vprime #Mest_k'(v')
        
        #set of points : || Mest_k(v) − Mest_k′ (v′)||
        tre = np.sqrt((v_in_world[0]-vprime_in_world[0])**2 + (v_in_world[1]-vprime_in_world[1])**2 + (v_in_world[2]-vprime_in_world[2])**2)
        set_tre[index]=tre
     
    return set_tre

def slice_tre(set_v : np.array,
             set_vprime : np.array ,
             stack_fk : 'list[SliceObject]',
             stack_fkprime : 'list[SliceObject]',
             Features_fk : 'list[sliceFeature]',
             Features_fkprime : 'list[sliceFeature]'
             ) -> np.array:
    

    """
    Take two volume with simulated motion and compute the mean TRE on each slice. The mean TRE is save in the sliceFeatures correscponding to slice k and k' respectively
    """

    nbpointError = set_v.shape[0]
    tre = np.zeros(nbpointError)
    
    for index in range(nbpointError):
        
        v = set_v[index,:]
        
        vprime = set_vprime[index,:]
        
        k = int(np.ceil(v[2]))
        kprime = int(np.ceil(vprime[2]))
        
        Mest_k =  stack_fk[k].get_estimatedTransfo()    
        homogenous_v = np.zeros(4)
        homogenous_v[0:2] = v[0:2]
        homogenous_v[3] = 1
        v_in_world = Mest_k @ homogenous_v #Mest_k(v)
                
        Mest_kprime = stack_fkprime[kprime].get_estimatedTransfo()
        homogenous_vprime = np.zeros(4)
        homogenous_vprime[0:2] = vprime[0:2]
        homogenous_vprime[3] = 1
        vprime_in_world = Mest_kprime @ homogenous_vprime #Mest_k'(v')
        
        #set of points : ||Mest_k(v) − Mest_k′(v′)||
        diff = np.sqrt((v_in_world[0]-vprime_in_world[0])**2 + (v_in_world[1]-vprime_in_world[1])**2 + (v_in_world[2]-vprime_in_world[2])**2)
                
        feature_k = Features_fk[k]

        #Computation of the mean tre for the slice is done here (inside the add_registration error): 
        #TRE_k = (1/card(k')) * sum_{k'}(||Mest_k(v) − Mest_k′(v′)||)
        feature_k.add_registration_error(diff) 
                 
        feature_kprime = Features_fkprime[kprime]

        #Computation of the mean tre for the slice is done here (inside the add_registration error): 
        #TRE_k = (1/card(k')) * sum_{k'}(||Mest_k(v) − Mest_k′(v′)||)
        feature_kprime.add_registration_error(diff)
        
        tre[index]=diff
      
    return tre

def same_order(listSlice,listnomvt,listFeature,transfo):
    
    img,_=separate_slices_in_stacks(listSlice)
    img=np.array(img,dtype=list)
    print(img.shape)
    nomvt,_=separate_slices_in_stacks(listnomvt)
    nomvt=np.array(nomvt,dtype=list)
    print(nomvt.shape)
    features=separate_features_in_stacks(listFeature)
    features=np.array(features,dtype=list)
    vectzimg = np.zeros(len(img))
    vectznomvt = np.zeros(len(img))

    
    #affine matrix are supposed to be the same, but in different order
    for image in range(0,len(img)):
            
            timg=img[image][0].get_slice().affine
            nximg=timg[0:3,0].copy()
            nyimg=timg[0:3,1].copy()
            nx=nximg/np.linalg.norm(nximg)
            ny=nyimg/np.linalg.norm(nyimg)
            nz=np.abs(np.cross(nx,ny))
            vectzimg[image]=np.argmax(nz)

            tnomvt=nomvt[image][0].get_slice().affine
            nxnomvt=tnomvt[0:3,0].copy()
            nynomvt=tnomvt[0:3,1].copy()
            nx=nxnomvt/np.linalg.norm(nxnomvt)
            ny=nynomvt/np.linalg.norm(nynomvt)
            nz=np.abs(np.cross(nx,ny))
            vectznomvt[image]=np.argmax(nz)


    minimg = np.argsort(vectzimg)
    minnomvt = np.argsort(vectznomvt)

    return img[minimg],nomvt[minnomvt],features[minimg],transfo[minnomvt]
       
def tre_for_each_slices(NoMotionSlices : 'list[SliceObject]',
                        listOfSlice : 'list[SliceObject]',
                        listFeatures : 'list[sliceFeature]',
                        transfo : np.array,
                        rejected_slice : list):
    """
    Compute the mean tre for each slices. Results are stored in sliceFeature corresponding to each slice
    """  
    images,nomvt,features,transfo = same_order(listOfSlice,NoMotionSlices,listFeatures,transfo)
    print(len(images[0]),len(nomvt[0]),len(images[1]),len(nomvt[1]),len(images[2]),len(nomvt[2])) 
    listOfSlice = np.concatenate(images)
    listOfSlice = listOfSlice.tolist()
    NoMotionSlices = np.concatenate(nomvt)
    NoMotionSlices = NoMotionSlices.tolist()
    listFeatures = np.concatenate(features)
    listFeatures = listFeatures.tolist()

    Features = separate_features_in_stacks(listFeatures) 
    #[error.reinitialized_error() for error in Features]

    NoMotionStacks,_ = separate_slices_in_stacks(NoMotionSlices.copy()) 
    Stacks,_ = separate_slices_in_stacks(listOfSlice.copy())
    print('Hahaha')
    print(len(Stacks),len(listOfSlice))

    for fk in range(len(Stacks)):
        for fkprime in range(len(Stacks)):
            print(fk,fkprime)
            if fk < fkprime:
               
               
               M_k = np.load(transfo[fk])
               M_kprime = np.load(transfo[fkprime])
               print(len(NoMotionStacks[fk]),len(Stacks[fk]),len(NoMotionStacks[fkprime]),len(Stacks[fkprime]))
 
               set_v, set_vprime = tre_indexes(NoMotionStacks[fk],NoMotionStacks[fkprime],M_k,M_kprime,rejected_slice) #common points between volumes when no movement
               print(set_vprime)
               slice_tre(set_v,set_vprime,Stacks[fk],Stacks[fkprime],Features[fk],Features[fkprime])


def distance_from_mask_edges(image_distance : np.array ,
                             set_v : np.array) -> np.array:
    """
    Associate each v point with it's chamfer distance from the mask edges.
    """
    
    distance = np.zeros(set_v.shape[0])
    indice = 0
    
    for v in set_v:    
        distance[indice] = image_distance[int(v[0]),int(v[1]),int(v[2])]
        indice = indice + 1
    
    return distance

    
def compute_image_distance(mask : Nifti1Image) -> np.array :
    """
    Compute the chamfer distance map with the border of the mask
    """
    
    inv_chamfer_distance = distance_transform_cdt(mask.get_fdata())

    return inv_chamfer_distance        
    
    