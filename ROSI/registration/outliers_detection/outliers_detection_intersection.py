#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 10:31:48 2022

@author: mercier
"""
import numpy as np 
#import registration
import scipy


class ErrorSlice: #object that contains all features for outliers classification
    
    def __init__(self,orientation,index_slice):
        self._orientation = orientation
        self._index = index_slice
        self._dice = 0
        self._mse = 0
        self._inter = 0
        self._error = 0
        self._nbpoint = 0
        self._slice_error = 0
        self._mask_proportion = 0
        self._bord = False
        self._std_intensity = 0
        self._frequency = 0
        self._center_distance = 0
        self._ncc=-1
        self._mask_point=0

    def add_registration_error(self,new_error):
        self._error=self._error+new_error
        self._nbpoint=self._nbpoint+1
        self._slice_error = self._error/self._nbpoint
     
    def get_error(self):
        return self._slice_error
    
    def set_error(self,er):
        self._slice_error = er
    
    def set_dice(self,dice):
        self._dice=dice
        
    def get_dice(self):
        return self._dice
    
    def set_mse(self,new_mse):
        self._mse=new_mse
        
    def get_mse(self):
        return self._mse
    
    def set_mask_proportion(self,pmask):
        self._mask_proportion = pmask
        
    def get_mask_proportion(self):
        return self._mask_proportion
        
    def set_bords(self,edge):
        self._bord = edge
        
    def edge(self):
        return self._bord
    
    def set_inter(self,delta):
        self._inter=delta
        
    def get_inter(self):
        return self._inter
    
    def get_orientation(self):
        return self._orientation
    
    def get_index(self):
        return self._index
    
    def get_nbpoint(self):
        return self._nbpoint
    
    def set_nbpoint(self,nb):
        self._nbpoint = nb

    def set_mask_point(self,nb):
        self._mask_point = nb
    
    def get_mask_point(self):
        return self._mask_point
    
    def set_std_intensity(self,std):
        self._std_intensity=std
    
    def get_std_intensity(self):
        return self._std_intensity
    
    def set_frequency(self,freq):
        self._frequency=freq
        
    def get_set_frequency(self):
        return self._frequency
    
    def set_center_distance(self,cd):
        self._center_distance=cd
    
    def get_center_distance(self):
        return self._center_distance
    
    def set_ncc(self,ncc_var):
        self._ncc = ncc_var
    
    def get_ncc(self):
        return self._ncc
        

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