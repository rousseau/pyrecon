#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 09:59:09 2021

@author: mercier
"""

import numpy as np
from .transformation import rotationCentre, rigidMatrix
from nibabel import Nifti1Image

"""
A sliceObject is a 2D image that
* belongs to a stack : self.__stack_index
* has an 2D binary segmetantion 
* has a position within a stack : self.__slice_index 
* has a particular orientation (e.g. axial) : self.__volume_index
* has a brain segmentation : self.__mask
* has an estimated position in the world coordinate system : self._estimated_transfo   
"""

class SliceObject: 
    def __init__(self,
                 sliceimage : Nifti1Image,
                 slicemask : np.array,
                 num_stack : int,
                 num_slice : int,
                 num_volume : int) :
        
        self.__2dslice = sliceimage  
        self.__parameters = np.zeros(6) #initial parameters of the rigid transformation, 3 for rotation and 3 for translation
        self.__rigid_matrix = rigidMatrix(self.__parameters) #initial rigid matrix : M(theta_k,t_k)
        slice_center_in_image = rotationCentre(slicemask) #The barycentre of the mask
        
        self.__corner_to_center = np.eye(4) 
        self.__center_to_corner = np.eye(4) 
        slice_transformation = self.__2dslice.affine #transformation matrix to convert the 2d slice into the world coordinate system
        center = slice_transformation @ slice_center_in_image #the barycentre of the mask, converted to world coordinate system : c_k
        self.__corner_to_center[0:3,3] = -center[0:3] #Translation, in world coordinates, from the slice corner (point (0,0) - the default value when applying a transformation) to the slice centre : T(-c_k)
        self.__center_to_corner[0:3,3] = +center[0:3] #the inverse translation : T(c_k)
        
        #The estimated transformation : 
        #M_k = T(c_k) M(theta_k,t_k) T(-c_k) R_f(k) R_k,2t3d
        self.__estimated_transfo = self.__center_to_corner @ (self.__rigid_matrix @ (self.__corner_to_center @ slice_transformation)) 
        

        self.__mask = slicemask #slice's mask
        self.__stack_index = num_stack #index of the stack, usually we use between 3 and 9 stacks
        self.__slice_index = num_slice #slice index within the stack, same as the z coordinate in the original LR image
        self.__volume_index = num_volume #index of the image - different stack can correspond to the same image, ie can have the same orientation (ex : two stack with coronnal orientation)

    #class methods 
    def get_parameters(self) -> np.array : #return parameters of the estimated rigid transformation
        return self.__parameters
    
    def get_stackIndex(self): #return stack index
        return self.__stack_index
    
    def set_parameters(self,x : np.array): #change parameters for the rigid matrix and estimate a new transformation
        self.__parameters = x
        self.__rigid_matrix = rigidMatrix(x)
        self.__estimated_transfo = self.__center_to_corner @ (self.__rigid_matrix @ (self.__corner_to_center @ self.__2dslice.affine))
        
    def get_estimatedTransfo(self) -> np.array : #return the estimated transformation
        return self.__estimated_transfo
    
    def get_slice(self): #return the 2d slice in a Nifty image. Be careful : the affine of the return image IS NOT the estimated matrix but slice_transformation
        return self.__2dslice
    
    def set_slice(self,newSlice : Nifti1Image): #change the slice
        self.__2dslice = newSlice
    
    def get_mask(self): #return the mask
        return self.__mask
    
    def get_centerOfRotation(self): #returns the centre of rotation used to apply the rigid matrix.
        return self.__corner_to_center[0:3,3]
    
    def get_indexSlice(self): #return the position of the slice within the stack
        return self.__slice_index
    
    def copy(self):
        copy = SliceObject(self.__2dslice, self.__mask, self.__stack_index,self.__slice_index,self.__volume_index)
        copy.set_parameters(self.__parameters)
        return copy
   
    def get_indexVolume(self): #return the stack index
        return self.__volume_index
    
