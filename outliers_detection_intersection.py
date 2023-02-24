#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 10:31:48 2022

@author: mercier
"""
import numpy as np 
import registration
import scipy

class ErrorSlice:
    
    def __init__(self,orientation,index_slice):
        self._orientation = orientation
        self._index = index_slice
        self._dice = 0
        self._mse = 10
        self._inter = 10
        self._error = 0
        self._nbpoint = 0
        self._slice_error = 0
        self._mask_proportion = 0
        self._bord = False
        
    def add_registration_error(self,new_error):
        self._error=self._error+new_error
        self._nbpoint=self._nbpoint+1
        self._slice_error = self._error/self._nbpoint
     
    def get_error(self):
        return self._slice_error
    
    def set_dice(self,dice):
        self._dice=dice
        
    def get_dice(self):
        return self._dice
    
    def set_mse(self,new_mse):
        self._mse=new_mse
        
    def set_mask_proportion(self,pmask):
        self._mask_proportion = pmask
        
    def set_bords(self,edge):
        self._bord = edge
    
    def get_mse(self):
        return self._mse
    
    def set_inter(self,delta):
        self._inter=delta
        
    def get_inter(self):
        return self._inter
    
    def get_orientation(self):
        return self._orientation
    
    def get_index(self):
        return self._index
    
    
    def get_mask_proportion(self):
        return self._mask_proportion
    
    def edge(self):
        return self._bord
 

def createVolumesFromAlistError(listError):
   
    """
    re-create the differents original stacks of the list (ex : Axial, Sagittal, Coronal)
    """
    
    orientation = []; listvolumeSliceError=[];

    for s in listError:
        
        s_or = s.get_orientation()#s.get_index_image()
        #print('sor',s_or)
        if s_or in orientation:
            #print('orientation',orientation)
            index_orientation = orientation.index(s_or)
            listvolumeSliceError[index_orientation].append(s)
        else:
            orientation.append(s_or)
            listvolumeSliceError.append([])
            index_orientation = orientation.index(s_or)
            listvolumeSliceError[index_orientation].append(s)
                
    return listvolumeSliceError   
 
    
 
    
#compute mask delineation coordinate in image
def mask_coordinate_in_image(val1,index1,nbpoint,pointImg1,pointImg2,coeff,pt,M1) : 
    
    distance1 = np.linalg.norm(pointImg1[0:2,1]-pointImg1[0:2,0])
    distance2 = np.linalg.norm(pointImg1[0:2,1]-pointImg1[0:2,0])
    dist = (max(distance1,distance2)+1)/nbpoint


    segment_coeff = np.linalg.inv(M1[0:3,0:3]) @ coeff
    point = np.concatenate((pt,np.array([1])))
    point_image = np.linalg.inv(M1) @ point
    #print("point_image ", point_image)

    
    for i in range(nbpoint) : 
        value = index1[i]
        if value==True:
            break
    

    mask_value_min = np.rint((i-1)*dist*segment_coeff[0:2] + pointImg1[0:2,0])

    mask_value_min = np.concatenate((mask_value_min,np.array([0,1])))
    
    for i in range(nbpoint) : 
        value = index1[-i]
        if value==True:
            break

    mask_value_max = np.rint((nbpoint-i-1)*dist*segment_coeff[0:2] + pointImg1[0:2,1])

    mask_value_max = np.concatenate((mask_value_max,np.array([0,1])))
    
    return mask_value_min,mask_value_max


#compute mask delineation coordinate in world
def mask_coordinate_in_mask(mask_value_min,mask_value_max,M,coeff,pt):
   
   interw = np.zeros((4,2)) #2 points of intersection, with 3 coordinates x,y,z
   interw[3,:] = np.ones((1,2))
   
   interw[:,0] = M @ mask_value_min
   interw[:,1] = M @ mask_value_max

   interw[0:3,0] = interw[0:3,0] - pt[0:3]
   interw[0:3,1] = interw[0:3,1] - pt[0:3]
   
   squareNorm = coeff @ coeff.transpose()
   coordinate_in_world = (1/squareNorm) * coeff.transpose() @ interw[0:3,:]
   return coordinate_in_world



def compute_distance(lambda_value,coeff,pt):
    
    point_in_world = np.zeros((4,2))
    point_in_world[0:3,0] = lambda_value[0] * coeff + pt
    point_in_world[0:3,1] = lambda_value[1] * coeff + pt
    distance = np.linalg.norm(point_in_world[0:3,0]-point_in_world[0:3,1])
    return distance
    

def compute_dice_error_in_mm(slice1,slice2):
    
    ratio = 0 
    distance = 0 
    #compute coordinate of intersection in two images
    sliceimage1=slice1.get_slice().get_fdata();M1=slice1.get_transfo();sliceimage2=slice2.get_slice().get_fdata();M2=slice2.get_transfo();res=min(slice1.get_slice().header.get_zooms())
    coeff,pt,ok = registration.intersectionLineBtw2Planes(M1,M2)
    pointImg1,pointImg2,nbpoint,ok = registration.commonSegment(sliceimage1,M1,sliceimage2,M2,res)
    ok=int(ok[0,0]); nbpoint=int(nbpoint[0,0])
    distance_union=0
    distance_intersection=0
    
    if ok==1:
        #compute slice profil along the intersection
        
        val1,index1,nbpointSlice1=registration.sliceProfil(slice1, pointImg1, nbpoint) 
        #print(nbpointSlice1)
        #profile in slice 1
        val2,index2,nbpointSlice2=registration.sliceProfil(slice2, pointImg2, nbpoint)
        #print(nbpointSlice2)
        
        if nbpointSlice1 == 0 and nbpointSlice2==0 : 
            distance_union=0
            distance_intersection=0
            distance=0
            return distance_union,distance_intersection,distance
        
        mask_value_min_slice1,mask_value_max_slice1 = mask_coordinate_in_image(val1,index1,nbpoint,pointImg1,pointImg2,coeff,pt,M1)
        #print(mask_value_min_slice1,mask_value_max_slice1)
        mask_value_min_slice2,mask_value_max_slice2 = mask_coordinate_in_image(val2,index2,nbpoint,pointImg2,pointImg1,coeff,pt,M2)    
        #print(mask_value_min_slice2,mask_value_max_slice2)
        
        world_mask_slice1 = mask_coordinate_in_mask(mask_value_min_slice1,mask_value_max_slice1,M1,coeff,pt)
        world_mask_slice2 = mask_coordinate_in_mask(mask_value_min_slice2,mask_value_max_slice2,M2,coeff,pt)
    
        #compute intersection distance (in mm) between the two mask 
        lambda_intersection = registration.minLambda(world_mask_slice1,world_mask_slice2)
        
        distance_intersection = compute_distance(lambda_intersection,coeff,pt)
        #print(distance_intersection)
    
        #compute union distance (in mm) between the two mask
        lambda_union = registration.minLambda(world_mask_slice1,world_mask_slice2,'union')

        distance_union = compute_distance(lambda_union,coeff,pt)
        #print(distance_union)
    
        distance=1
        
    return distance_union,distance_intersection,distance



def compute_dice_error_for_each_slices(union,intersection,nb_slice_matrix,listSlice):
    
    for i_slice1 in range(len(listSlice)):
        for i_slice2 in range(len(listSlice)):
            if i_slice1>i_slice2 : 
                    slice1 = listSlice[i_slice1]
                    slice2 = listSlice[i_slice2]
                    if slice1.get_orientation() != slice2.get_orientation():
                        union[i_slice1,i_slice2],intersection[i_slice1,i_slice2],nb_slice_matrix[i_slice1,i_slice2] = compute_dice_error_in_mm(slice1,slice2)
               
            
    return union,intersection,nb_slice_matrix


def intersection_profil_on_mask(Slice,pointImg,nbpoint) :
    
    if nbpoint == 0:
        return 0,0,0

    value_intersection = np.zeros(nbpoint)
    pointInterpol = np.zeros((3,nbpoint))
    pointInterpol[0,:] = np.linspace(pointImg[0,0],pointImg[0,1],nbpoint)
    pointInterpol[1,:] = np.linspace(pointImg[1,0],pointImg[1,1],nbpoint)
   
    
    mask = Slice.get_mask()
    scipy.ndimage.map_coordinates(mask, pointInterpol, output=value_intersection, order=0, mode='constant',cval=0,prefilter=False)
    

    return value_intersection


def get_index(orientation,index_slice,listSlice):
    index_list = -1
    for i_slice in range(0,len(listSlice)):
        slicei = listSlice[i_slice]
        if slicei.get_orientation() == orientation and slicei.get_index_slice()==index_slice:
            index_list = listSlice.index(slicei)
            break
    return index_list




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